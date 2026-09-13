#include "onnxruntime_c_api.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/utsname.h>
#include <unistd.h>
#include <sched.h>
#include <utility>
#include <vector>

namespace {

const OrtApi* api = nullptr;

void Check(OrtStatus* status) {
  if (status == nullptr) return;
  std::string message = api->GetErrorMessage(status);
  api->ReleaseStatus(status);
  throw std::runtime_error(message);
}

std::string Json(const std::string& value) {
  std::ostringstream output;
  output << '"';
  for (unsigned char character : value) {
    switch (character) {
      case '"': output << "\\\""; break;
      case '\\': output << "\\\\"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (character < 0x20) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<int>(character) << std::dec;
        } else {
          output << character;
        }
    }
  }
  output << '"';
  return output.str();
}

uint32_t ReadU32(std::istream& stream) {
  unsigned char bytes[4];
  stream.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
  if (!stream) throw std::runtime_error("truncated initializer sidecar");
  return static_cast<uint32_t>(bytes[0]) |
         (static_cast<uint32_t>(bytes[1]) << 8) |
         (static_cast<uint32_t>(bytes[2]) << 16) |
         (static_cast<uint32_t>(bytes[3]) << 24);
}

uint64_t ReadU64(std::istream& stream) {
  unsigned char bytes[8];
  stream.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
  if (!stream) throw std::runtime_error("truncated initializer sidecar");
  uint64_t result = 0;
  for (int index = 7; index >= 0; --index) result = (result << 8) | bytes[index];
  return result;
}

struct Initializer {
  std::string name;
  ONNXTensorElementDataType type{};
  std::vector<int64_t> shape;
  std::vector<unsigned char> data;
  OrtValue* value = nullptr;
};

std::vector<Initializer> ReadInitializers(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open initializer sidecar: " + path);
  char magic[8];
  stream.read(magic, sizeof(magic));
  if (!stream || std::memcmp(magic, "FIBERPW1", sizeof(magic)) != 0) {
    throw std::runtime_error("invalid initializer sidecar magic");
  }
  const uint32_t count = ReadU32(stream);
  std::vector<Initializer> result;
  result.reserve(count);
  for (uint32_t index = 0; index < count; ++index) {
    const uint32_t name_size = ReadU32(stream);
    const uint32_t element_type = ReadU32(stream);
    const uint32_t rank = ReadU32(stream);
    const uint64_t data_size = ReadU64(stream);
    Initializer item;
    item.name.resize(name_size);
    stream.read(item.name.data(), name_size);
    item.type = static_cast<ONNXTensorElementDataType>(element_type);
    item.shape.reserve(rank);
    for (uint32_t dimension = 0; dimension < rank; ++dimension) {
      item.shape.push_back(static_cast<int64_t>(ReadU64(stream)));
    }
    item.data.resize(data_size);
    stream.read(reinterpret_cast<char*>(item.data.data()), static_cast<std::streamsize>(data_size));
    if (!stream) throw std::runtime_error("truncated initializer sidecar payload");
    result.push_back(std::move(item));
  }
  if (stream.peek() != std::char_traits<char>::eof()) {
    throw std::runtime_error("trailing bytes in initializer sidecar");
  }
  return result;
}

struct Options {
  std::string model;
  std::string initializers;
  std::string output;
  std::vector<std::string> custom_ops;
  std::string cpu = "auto";
  int warmup = 500;
  int iterations = 5000;
  int repeats = 5;
  int seed = 1;
  double frame_ms = 16.0;
  bool ort_info = false;
  bool embedded_initializers = false;
};

Options Parse(int argc, char** argv) {
  Options result;
  for (int index = 1; index < argc; ++index) {
    const std::string key = argv[index];
    if (key == "--ort-info") {
      result.ort_info = true;
      continue;
    }
    if (key == "--embedded-initializers") {
      result.embedded_initializers = true;
      continue;
    }
    if (index + 1 >= argc) throw std::runtime_error("missing value for " + key);
    const std::string value = argv[++index];
    if (key == "--model") result.model = value;
    else if (key == "--initializers") result.initializers = value;
    else if (key == "--output") result.output = value;
    else if (key == "--custom-op") result.custom_ops.push_back(value);
    else if (key == "--cpu") result.cpu = value;
    else if (key == "--warmup") result.warmup = std::stoi(value);
    else if (key == "--iterations") result.iterations = std::stoi(value);
    else if (key == "--repeats") result.repeats = std::stoi(value);
    else if (key == "--seed") result.seed = std::stoi(value);
    else if (key == "--frame-ms") result.frame_ms = std::stod(value);
    else throw std::runtime_error("unknown argument: " + key);
  }
  if (result.model.empty() || result.output.empty() ||
      (!result.embedded_initializers && result.initializers.empty())) {
    throw std::runtime_error(
        "--model, --output, and either --initializers or "
        "--embedded-initializers are required");
  }
  if (result.warmup < 0 || result.iterations < 1 || result.repeats < 1) {
    throw std::runtime_error("warmup must be non-negative; iterations/repeats must be positive");
  }
  return result;
}

int SetAffinity(const std::string& requested) {
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
    throw std::runtime_error("sched_getaffinity failed");
  }
  int selected = -1;
  if (requested == "auto") {
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
      if (CPU_ISSET(cpu, &allowed)) { selected = cpu; break; }
    }
  } else {
    selected = std::stoi(requested);
  }
  if (selected < 0 || selected >= CPU_SETSIZE || !CPU_ISSET(selected, &allowed)) {
    throw std::runtime_error("requested CPU is outside current affinity");
  }
  cpu_set_t target;
  CPU_ZERO(&target);
  CPU_SET(selected, &target);
  if (sched_setaffinity(0, sizeof(target), &target) != 0) {
    throw std::runtime_error("sched_setaffinity failed");
  }
  return selected;
}

std::string CpuName() {
  std::ifstream stream("/proc/cpuinfo");
  std::string line;
  while (std::getline(stream, line)) {
    if (line.rfind("model name", 0) == 0 || line.rfind("Hardware", 0) == 0 ||
        line.rfind("Model name", 0) == 0) {
      const auto separator = line.find(':');
      if (separator != std::string::npos) {
        const auto start = line.find_first_not_of(" \t", separator + 1);
        return start == std::string::npos ? "unknown" : line.substr(start);
      }
    }
  }
  return "unknown";
}

std::string Hostname() {
  char value[256]{};
  if (gethostname(value, sizeof(value) - 1) != 0) return "unknown";
  return value;
}

std::string Machine() {
  struct utsname value {};
  if (uname(&value) != 0) return "unknown";
  return value.machine;
}

struct TensorSpec {
  std::string name;
  std::vector<int64_t> shape;
};

std::vector<TensorSpec> GetSpecs(OrtSession* session, bool inputs) {
  size_t count = 0;
  Check(inputs ? api->SessionGetInputCount(session, &count)
               : api->SessionGetOutputCount(session, &count));
  OrtAllocator* allocator = nullptr;
  Check(api->GetAllocatorWithDefaultOptions(&allocator));
  std::vector<TensorSpec> result;
  for (size_t index = 0; index < count; ++index) {
    char* name = nullptr;
    Check(inputs ? api->SessionGetInputName(session, index, allocator, &name)
                 : api->SessionGetOutputName(session, index, allocator, &name));
    OrtTypeInfo* type_info = nullptr;
    Check(inputs ? api->SessionGetInputTypeInfo(session, index, &type_info)
                 : api->SessionGetOutputTypeInfo(session, index, &type_info));
    const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
    Check(api->CastTypeInfoToTensorInfo(type_info, &tensor_info));
    ONNXTensorElementDataType type{};
    Check(api->GetTensorElementType(tensor_info, &type));
    if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      api->ReleaseTypeInfo(type_info);
      allocator->Free(allocator, name);
      throw std::runtime_error("runner supports float32 graph inputs and outputs only");
    }
    size_t rank = 0;
    Check(api->GetDimensionsCount(tensor_info, &rank));
    TensorSpec spec;
    spec.name = name;
    spec.shape.resize(rank);
    Check(api->GetDimensions(tensor_info, spec.shape.data(), rank));
    if (std::any_of(spec.shape.begin(), spec.shape.end(), [](int64_t value) { return value < 0; })) {
      api->ReleaseTypeInfo(type_info);
      allocator->Free(allocator, name);
      throw std::runtime_error("runner requires static input/output shapes");
    }
    api->ReleaseTypeInfo(type_info);
    allocator->Free(allocator, name);
    result.push_back(std::move(spec));
  }
  return result;
}

size_t Elements(const std::vector<int64_t>& shape) {
  return std::accumulate(shape.begin(), shape.end(), size_t{1},
                         [](size_t left, int64_t right) { return left * static_cast<size_t>(right); });
}

struct TensorBuffer {
  std::vector<float> data;
  OrtValue* value = nullptr;
};

TensorBuffer MakeTensor(const TensorSpec& spec, OrtMemoryInfo* memory) {
  TensorBuffer result;
  result.data.resize(Elements(spec.shape));
  Check(api->CreateTensorWithDataAsOrtValue(
      memory, result.data.data(), result.data.size() * sizeof(float), spec.shape.data(),
      spec.shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &result.value));
  return result;
}

double Quantile(std::vector<double> values, double probability) {
  std::sort(values.begin(), values.end());
  const double position = probability * static_cast<double>(values.size() - 1);
  const size_t lower = static_cast<size_t>(std::floor(position));
  const size_t upper = static_cast<size_t>(std::ceil(position));
  const double fraction = position - static_cast<double>(lower);
  return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

double Mean(const std::vector<double>& values) {
  return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

void ReleaseBuffers(std::vector<TensorBuffer>& values) {
  for (auto& value : values) api->ReleaseValue(value.value);
}

}

int main(int argc, char** argv) {
  try {
    const Options options = Parse(argc, argv);
    api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (api == nullptr) throw std::runtime_error("failed to acquire ORT API");
    const int selected_cpu = SetAffinity(options.cpu);

    OrtEnv* env = nullptr;
    Check(api->CreateEnv(options.ort_info ? ORT_LOGGING_LEVEL_INFO : ORT_LOGGING_LEVEL_WARNING,
                         "fiber-optimized-deployment", &env));
    OrtSessionOptions* session_options = nullptr;
    Check(api->CreateSessionOptions(&session_options));
    Check(api->SetIntraOpNumThreads(session_options, 1));
    Check(api->SetInterOpNumThreads(session_options, 1));
    Check(api->SetSessionExecutionMode(session_options, ORT_SEQUENTIAL));
    Check(api->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_ALL));
    Check(api->AddSessionConfigEntry(session_options, "session.intra_op.allow_spinning", "0"));
    Check(api->AddSessionConfigEntry(session_options, "session.inter_op.allow_spinning", "0"));
    for (const auto& library : options.custom_ops) {
      Check(api->RegisterCustomOpsLibrary_V2(session_options, library.c_str()));
    }

    OrtMemoryInfo* memory = nullptr;
    Check(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory));
    auto initializers = options.embedded_initializers
        ? std::vector<Initializer>{}
        : ReadInitializers(options.initializers);
    uint64_t initializer_bytes = 0;
    for (auto& initializer : initializers) {
      initializer_bytes += initializer.data.size();
      Check(api->CreateTensorWithDataAsOrtValue(
          memory, initializer.data.data(), initializer.data.size(), initializer.shape.data(),
          initializer.shape.size(), initializer.type, &initializer.value));
      Check(api->AddInitializer(session_options, initializer.name.c_str(), initializer.value));
    }

    OrtPrepackedWeightsContainer* container = nullptr;
    Check(api->CreatePrepackedWeightsContainer(&container));
    OrtSession* prime = nullptr;
    Check(api->CreateSessionWithPrepackedWeightsContainer(
        env, options.model.c_str(), session_options, container, &prime));
    OrtSession* session = nullptr;
    Check(api->CreateSessionWithPrepackedWeightsContainer(
        env, options.model.c_str(), session_options, container, &session));
    api->ReleaseSession(prime);
    api->ReleaseSessionOptions(session_options);

    const auto input_specs = GetSpecs(session, true);
    const auto output_specs = GetSpecs(session, false);
    if (input_specs.size() != output_specs.size() || input_specs.size() < 2) {
      throw std::runtime_error("expected one frame plus matching explicit states");
    }

    TensorBuffer frame = MakeTensor(input_specs[0], memory);
    std::vector<TensorBuffer> output_frames;
    output_frames.push_back(MakeTensor(output_specs[0], memory));
    output_frames.push_back(MakeTensor(output_specs[0], memory));
    std::vector<TensorBuffer> state_banks[2];
    for (int bank = 0; bank < 2; ++bank) {
      for (size_t index = 1; index < input_specs.size(); ++index) {
        if (input_specs[index].shape != output_specs[index].shape) {
          throw std::runtime_error("state input/output shape mismatch");
        }
        state_banks[bank].push_back(MakeTensor(input_specs[index], memory));
      }
    }

    OrtIoBinding* bindings[2]{};
    for (int source = 0; source < 2; ++source) {
      const int target = 1 - source;
      Check(api->CreateIoBinding(session, &bindings[source]));
      Check(api->BindInput(bindings[source], input_specs[0].name.c_str(), frame.value));
      for (size_t index = 1; index < input_specs.size(); ++index) {
        Check(api->BindInput(bindings[source], input_specs[index].name.c_str(),
                             state_banks[source][index - 1].value));
      }
      Check(api->BindOutput(bindings[source], output_specs[0].name.c_str(),
                            output_frames[source].value));
      for (size_t index = 1; index < output_specs.size(); ++index) {
        Check(api->BindOutput(bindings[source], output_specs[index].name.c_str(),
                              state_banks[target][index - 1].value));
      }
    }
    OrtRunOptions* run_options = nullptr;
    Check(api->CreateRunOptions(&run_options));

    std::mt19937 generator(static_cast<uint32_t>(options.seed));
    std::normal_distribution<float> distribution(0.0F, 0.1F);
    std::vector<std::vector<float>> frames(256, std::vector<float>(frame.data.size()));
    for (auto& values : frames) {
      for (auto& value : values) value = distribution(generator);
    }
    int binding_index = 0;
    auto reset = [&]() {
      for (auto& bank : state_banks) {
        for (auto& state : bank) std::fill(state.data.begin(), state.data.end(), 0.0F);
      }
      binding_index = 0;
    };
    size_t frame_index = 0;
    auto step = [&]() -> double {
      std::copy(frames[frame_index % frames.size()].begin(),
                frames[frame_index % frames.size()].end(), frame.data.begin());
      ++frame_index;
      const auto started = std::chrono::steady_clock::now();
      Check(api->RunWithBinding(session, run_options, bindings[binding_index]));
      const auto stopped = std::chrono::steady_clock::now();
      binding_index = 1 - binding_index;
      return std::chrono::duration<double, std::milli>(stopped - started).count();
    };

    std::vector<std::vector<double>> repeat_values;
    std::vector<double> repeat_means;
    std::vector<double> combined;
    for (int repeat = 0; repeat < options.repeats; ++repeat) {
      reset();
      for (int index = 0; index < options.warmup; ++index) step();
      std::vector<double> values;
      values.reserve(options.iterations);
      for (int index = 0; index < options.iterations; ++index) values.push_back(step());
      repeat_means.push_back(Mean(values));
      combined.insert(combined.end(), values.begin(), values.end());
      repeat_values.push_back(values);
      std::cout << "{\"repeat\":" << repeat + 1 << ",\"mean_ms\":" << Mean(values)
                << ",\"p50_ms\":" << Quantile(values, 0.50)
                << ",\"p90_ms\":" << Quantile(values, 0.90)
                << ",\"p99_ms\":" << Quantile(values, 0.99) << "}" << std::endl;
    }

    double sample_std = 0.0;
    if (repeat_means.size() > 1) {
      const double average = Mean(repeat_means);
      for (double value : repeat_means) sample_std += (value - average) * (value - average);
      sample_std = std::sqrt(sample_std / static_cast<double>(repeat_means.size() - 1));
    }

    std::ofstream report(options.output);
    if (!report) throw std::runtime_error("cannot write report: " + options.output);
    report << std::setprecision(12)
           << "{\n  \"model\": " << Json(options.model)
           << ",\n  \"initializer_sidecar\": " << Json(options.initializers)
           << ",\n  \"initializer_storage\": "
           << Json(options.embedded_initializers ? "embedded" : "external-sidecar")
           << ",\n  \"runner\": \"cpp-optimized-deployment\""
           << ",\n  \"host\": " << Json(Hostname())
           << ",\n  \"cpu_model\": " << Json(CpuName())
           << ",\n  \"machine\": " << Json(Machine())
           << ",\n  \"selected_cpu\": " << selected_cpu
           << ",\n  \"onnxruntime_version\": " << Json(OrtGetApiBase()->GetVersionString())
           << ",\n  \"initializer_count\": " << initializers.size()
           << ",\n  \"initializer_bytes\": " << initializer_bytes
           << ",\n  \"prepacked_weights_container\": true"
           << ",\n  \"priming_sessions\": 1"
           << ",\n  \"measured_sessions\": 1"
           << ",\n  \"protocol\": {"
           << "\n    \"execution_provider\": \"CPUExecutionProvider\","
           << "\n    \"intra_op_threads\": 1,"
           << "\n    \"inter_op_threads\": 1,"
           << "\n    \"execution_mode\": \"sequential\","
           << "\n    \"graph_optimization\": \"all\","
           << "\n    \"io_binding\": true,"
           << "\n    \"continuous_state\": true,"
           << "\n    \"shared_initializer_bank_cache\": true,"
           << "\n    \"frame_duration_ms\": " << options.frame_ms << ','
           << "\n    \"warmup\": " << options.warmup << ','
           << "\n    \"iterations\": " << options.iterations << ','
           << "\n    \"repeats\": " << options.repeats << ','
           << "\n    \"seed\": " << options.seed << "\n  },"
           << "\n  \"interface\": {\n    \"input_names\": [";
    for (size_t index = 0; index < input_specs.size(); ++index) {
      if (index) report << ',';
      report << Json(input_specs[index].name);
    }
    report << "],\n    \"output_names\": [";
    for (size_t index = 0; index < output_specs.size(); ++index) {
      if (index) report << ',';
      report << Json(output_specs[index].name);
    }
    report << "]\n  },\n  \"latency\": {"
           << "\n    \"mean_ms\": " << Mean(repeat_means) << ','
           << "\n    \"median_repeat_mean_ms\": " << Quantile(repeat_means, 0.50) << ','
           << "\n    \"p50_ms\": " << Quantile(combined, 0.50) << ','
           << "\n    \"p90_ms\": " << Quantile(combined, 0.90) << ','
           << "\n    \"p99_ms\": " << Quantile(combined, 0.99) << ','
           << "\n    \"repeat_mean_sample_std_ms\": " << sample_std << ','
           << "\n    \"real_time_factor\": " << Mean(repeat_means) / options.frame_ms << ','
           << "\n    \"repeats\": [";
    for (size_t index = 0; index < repeat_values.size(); ++index) {
      if (index) report << ',';
      report << "\n      {\"repeat\": " << index + 1
             << ", \"mean_ms\": " << Mean(repeat_values[index])
             << ", \"p50_ms\": " << Quantile(repeat_values[index], 0.50)
             << ", \"p90_ms\": " << Quantile(repeat_values[index], 0.90)
             << ", \"p99_ms\": " << Quantile(repeat_values[index], 0.99) << '}';
    }
    report << "\n    ]\n  }\n}\n";

    api->ReleaseRunOptions(run_options);
    api->ReleaseIoBinding(bindings[0]);
    api->ReleaseIoBinding(bindings[1]);
    api->ReleaseValue(frame.value);
    ReleaseBuffers(output_frames);
    ReleaseBuffers(state_banks[0]);
    ReleaseBuffers(state_banks[1]);
    api->ReleaseSession(session);
    api->ReleasePrepackedWeightsContainer(container);
    for (auto& initializer : initializers) api->ReleaseValue(initializer.value);
    api->ReleaseMemoryInfo(memory);
    api->ReleaseEnv(env);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "benchmark_deployment: " << error.what() << std::endl;
    return 1;
  }
}

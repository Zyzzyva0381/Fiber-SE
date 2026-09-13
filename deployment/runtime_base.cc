#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "onnxruntime_lite_custom_op.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#ifndef SETRAIN_ARM_FIBRE_GRU_USE_ACL
#define SETRAIN_ARM_FIBRE_GRU_USE_ACL 0
#endif

#ifndef SETRAIN_ARM_FIBRE_GRU_USE_KLEIDIAI
#define SETRAIN_ARM_FIBRE_GRU_USE_KLEIDIAI 0
#endif

#if SETRAIN_ARM_FIBRE_GRU_USE_ACL && SETRAIN_ARM_FIBRE_GRU_USE_KLEIDIAI
#error Select at most one external FibreGRU matrix backend
#endif

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#if SETRAIN_ARM_FIBRE_GRU_USE_ACL
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/NEON/functions/NEGEMM.h"
#include "arm_compute/runtime/Tensor.h"
#endif

#if SETRAIN_ARM_FIBRE_GRU_USE_KLEIDIAI
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_x32p16x1bx32_x32_x32_neon.h"
#endif

#ifndef SETRAIN_ARM_REUSE_GRAM_POW
#define SETRAIN_ARM_REUSE_GRAM_POW 1
#endif

#ifndef SETRAIN_ARM_OPTIMIZED_FIBRE_LAYOUT
#define SETRAIN_ARM_OPTIMIZED_FIBRE_LAYOUT 1
#endif

namespace {

constexpr char kDomain[] = "com.setrain.cpu";
constexpr int64_t kInputFrequencies = 257;
constexpr int64_t kBodyFrequencies = 256;
constexpr int64_t kObservationChannels = 3;
constexpr int64_t kObservationOutputFrequencies = 64;
constexpr int64_t kObservationKernelSize = 9;
constexpr int64_t kObservationStride = 4;
constexpr int64_t kObservationPadding = 3;

#if SETRAIN_ARM_FIBRE_GRU_USE_ACL
struct FibreGruAclAffine {
  static constexpr int kHidden = 144;
  static constexpr int kBatch = 4;
  static constexpr int kGates = 3 * kHidden;

  FibreGruAclAffine(const float* input_weight, const float* recurrent_weight) {
    using arm_compute::GEMMInfo;
    arm_compute::NEScheduler::get().set_num_threads(1);
    Initialize(input_, input_matrix_, input_output_);
    Initialize(recurrent_, recurrent_matrix_, recurrent_output_);
    const GEMMInfo info(false, false, true, 0, false, true);
    input_gemm_.configure(
        &input_, &input_matrix_, nullptr, &input_output_, 1.0F, 0.0F, info);
    recurrent_gemm_.configure(
        &recurrent_, &recurrent_matrix_, nullptr, &recurrent_output_, 1.0F,
        0.0F, info);
    Allocate(input_, input_matrix_, input_output_);
    Allocate(recurrent_, recurrent_matrix_, recurrent_output_);
    TransposeWeight(input_weight, input_matrix_);
    TransposeWeight(recurrent_weight, recurrent_matrix_);
    input_gemm_.prepare();
    recurrent_gemm_.prepare();
    input_matrix_.allocator()->free();
    recurrent_matrix_.allocator()->free();
  }

  std::pair<const float*, const float*> Run(
      const float* input, const float* recurrent) {
    std::memcpy(input_.buffer(), input, kBatch * kHidden * sizeof(float));
    std::memcpy(
        recurrent_.buffer(), recurrent, kBatch * kHidden * sizeof(float));
    input_gemm_.run();
    recurrent_gemm_.run();
    return {
        reinterpret_cast<const float*>(input_output_.buffer()),
        reinterpret_cast<const float*>(recurrent_output_.buffer())};
  }

 private:
  static void Initialize(
      arm_compute::Tensor& values, arm_compute::Tensor& weight,
      arm_compute::Tensor& output) {
    using arm_compute::DataType;
    using arm_compute::TensorInfo;
    using arm_compute::TensorShape;
    values.allocator()->init(
        TensorInfo(TensorShape(kHidden, kBatch), 1, DataType::F32));
    weight.allocator()->init(
        TensorInfo(TensorShape(kGates, kHidden), 1, DataType::F32));
    output.allocator()->init(
        TensorInfo(TensorShape(kGates, kBatch), 1, DataType::F32));
  }

  static void Allocate(
      arm_compute::Tensor& values, arm_compute::Tensor& weight,
      arm_compute::Tensor& output) {
    values.allocator()->allocate();
    weight.allocator()->allocate();
    output.allocator()->allocate();
  }

  static void TransposeWeight(
      const float* source, arm_compute::Tensor& destination) {
    float* target = reinterpret_cast<float*>(destination.buffer());
    for (int input = 0; input < kHidden; ++input) {
      for (int output = 0; output < kGates; ++output) {
        target[input * kGates + output] = source[output * kHidden + input];
      }
    }
  }

  arm_compute::Tensor input_;
  arm_compute::Tensor input_matrix_;
  arm_compute::Tensor input_output_;
  arm_compute::Tensor recurrent_;
  arm_compute::Tensor recurrent_matrix_;
  arm_compute::Tensor recurrent_output_;
  arm_compute::NEGEMM input_gemm_;
  arm_compute::NEGEMM recurrent_gemm_;
};
#endif

#if SETRAIN_ARM_FIBRE_GRU_USE_KLEIDIAI
struct KleidiAiMatmulKernelArgs {
  float maxval;
  float minval;
  unsigned int num_strings;
  const unsigned int* string_lengths;
  size_t n;
  const void* packed_rhs;
  size_t output_offset;
  size_t input_initial_col;
  size_t input_offset;
  void* output;
  const void* bias;
};

extern "C" void
kai_kernel_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla(
    const void* input, size_t rows, KleidiAiMatmulKernelArgs* arguments,
    unsigned long flags);

template <int Hidden, int Batch>
struct FibreGruKaiAffine {
  static constexpr int kHidden = Hidden;
  static constexpr int kBatch = Batch;
  static constexpr int kGates = 3 * kHidden;

  FibreGruKaiAffine(const float* input_weight, const float* recurrent_weight) {
    const size_t packed_size =
        kai_get_rhs_packed_size_rhs_pack_nxk_x32p16x1bx32_x32_x32_neon(
            kGates, kHidden);
    input_packed_.resize(packed_size);
    recurrent_packed_.resize(packed_size);
    Pack(input_weight, input_packed_.data());
    Pack(recurrent_weight, recurrent_packed_.data());
    input_output_.resize(kBatch * kGates);
    recurrent_output_.resize(kBatch * kGates);
  }

  std::pair<const float*, const float*> Run(
      const float* input, const float* recurrent) {
    Multiply(input, input_packed_.data(), input_output_.data());
    Multiply(recurrent, recurrent_packed_.data(), recurrent_output_.data());
    return {input_output_.data(), recurrent_output_.data()};
  }

  std::pair<const float*, const float*> RunSplitInput(
      const float* const* first_rows, size_t first_columns,
      const float* const* second_rows, size_t second_columns,
      const float* recurrent) {
    MultiplyIndirect(
        first_rows, first_columns, second_rows, second_columns,
        input_packed_.data(), input_output_.data());
    Multiply(recurrent, recurrent_packed_.data(), recurrent_output_.data());
    return {input_output_.data(), recurrent_output_.data()};
  }

 private:
  static void Pack(const float* source, void* output) {
    const size_t nr =
        kai_get_nr_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla();
    const size_t kr =
        kai_get_kr_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla();
    const size_t sr =
        kai_get_sr_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla();
    kai_run_rhs_pack_nxk_x32p16x1bx32_x32_x32_neon(
        1, kGates, kHidden, nr, kr, sr, kHidden * sizeof(float), source,
        nullptr, nullptr, output, 0, nullptr);
  }

  static void Multiply(
      const float* input, const void* packed_weight, float* output) {
    constexpr int kRowsPerCall = 6;
    for (int row = 0; row < kBatch; row += kRowsPerCall) {
      const int rows = std::min(kRowsPerCall, kBatch - row);
      kai_run_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla(
          rows, kGates, kHidden, input + row * kHidden,
          kHidden * sizeof(float), packed_weight, output + row * kGates,
          kGates * sizeof(float), sizeof(float),
          -std::numeric_limits<float>::infinity(),
          std::numeric_limits<float>::infinity());
    }
  }

  static void MultiplyIndirect(
      const float* const* first_rows, size_t first_columns,
      const float* const* second_rows, size_t second_columns,
      const void* packed_weight, float* output) {
    const unsigned int lengths[2] = {
        static_cast<unsigned int>(first_columns),
        static_cast<unsigned int>(second_columns)};
    const void* strings[2] = {first_rows, second_rows};
    KleidiAiMatmulKernelArgs arguments{};
    arguments.maxval = std::numeric_limits<float>::infinity();
    arguments.minval = -std::numeric_limits<float>::infinity();
    arguments.num_strings = 2;
    arguments.string_lengths = lengths;
    arguments.n = kGates;
    arguments.packed_rhs = packed_weight;
    arguments.output_offset = kGates;
    arguments.input_initial_col = 0;
    arguments.input_offset = 0;
    arguments.output = output;
    arguments.bias = nullptr;
    constexpr unsigned long kFlagClamp = 0x2;
    constexpr unsigned long kFlagIndirectInput = 0x8;
    kai_kernel_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla(
        strings, kBatch, &arguments, kFlagClamp | kFlagIndirectInput);
  }

  std::vector<uint8_t> input_packed_;
  std::vector<uint8_t> recurrent_packed_;
  std::vector<float> input_output_;
  std::vector<float> recurrent_output_;
};

template <int Hidden, int Batch>
struct FibreProjectionKai {
  FibreProjectionKai(const float* weight, const float* bias) {
    std::vector<float> transposed(Hidden * Hidden);
    for (int input = 0; input < Hidden; ++input) {
      for (int output = 0; output < Hidden; ++output) {
        transposed[output * Hidden + input] = weight[input * Hidden + output];
      }
    }
    const size_t packed_size =
        kai_get_rhs_packed_size_rhs_pack_nxk_x32p16x1bx32_x32_x32_neon(
            Hidden, Hidden);
    packed_.resize(packed_size);
    const size_t nr =
        kai_get_nr_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla();
    const size_t kr =
        kai_get_kr_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla();
    const size_t sr =
        kai_get_sr_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla();
    kai_run_rhs_pack_nxk_x32p16x1bx32_x32_x32_neon(
        1, Hidden, Hidden, nr, kr, sr, Hidden * sizeof(float),
        transposed.data(), bias, nullptr, packed_.data(), 0, nullptr);
    output_.resize(Batch * Hidden);
  }

  const float* Run(const float* input) {
    kai_run_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla(
        Batch, Hidden, Hidden, input, Hidden * sizeof(float), packed_.data(),
        output_.data(), Hidden * sizeof(float), sizeof(float),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity());
    return output_.data();
  }

 private:
  std::vector<uint8_t> packed_;
  std::vector<float> output_;
};
#endif

uint64_t HashFloats(const float* data, size_t count) {
  uint64_t value = 1469598103934665603ULL;
  const auto* bytes = reinterpret_cast<const uint8_t*>(data);
  for (size_t index = 0; index < count * sizeof(float); ++index) {
    value ^= bytes[index];
    value *= 1099511628211ULL;
  }
  return value;
}

struct FibreGruBank {
  int64_t hidden{};
  std::vector<float> input;
  std::vector<float> recurrent;
#if SETRAIN_ARM_FIBRE_GRU_USE_ACL
  std::shared_ptr<FibreGruAclAffine> acl;
#endif
#if SETRAIN_ARM_FIBRE_GRU_USE_KLEIDIAI
  std::shared_ptr<FibreGruKaiAffine<20, 16>> kai_h20;
  std::shared_ptr<FibreGruKaiAffine<36, 24>> kai_h36;
  std::shared_ptr<FibreGruKaiAffine<48, 36>> kai_h48;
  std::shared_ptr<FibreGruKaiAffine<80, 4>> kai_h80;
  std::shared_ptr<FibreGruKaiAffine<144, 4>> kai_h144;
  std::shared_ptr<FibreGruKaiAffine<160, 4>> kai_h160_b4;
  std::shared_ptr<FibreGruKaiAffine<160, 8>> kai_h160_b8;
  std::shared_ptr<FibreGruKaiAffine<176, 6>> kai_h176;
#endif
};

std::atomic<size_t>& FibreGruPackedBankCount() {
  static std::atomic<size_t> count{0};
  return count;
}

std::atomic<size_t>& FibreGruPackedBankCacheHits() {
  static std::atomic<size_t> count{0};
  return count;
}

std::shared_ptr<const FibreGruBank> PackFibreGruBank(
    const float* input_weight, const float* recurrent_weight,
    int64_t hidden, int64_t batch_hint = 0) {
  const size_t matrix_elements = static_cast<size_t>(3 * hidden * hidden);
  uint64_t key = HashFloats(input_weight, matrix_elements);
  key ^= HashFloats(recurrent_weight, matrix_elements) +
         0x9e3779b97f4a7c15ULL + (key << 6) + (key >> 2);
  key ^= static_cast<uint64_t>(hidden) * 0x9e3779b97f4a7c15ULL;
  key ^= static_cast<uint64_t>(batch_hint) * 0xbf58476d1ce4e5b9ULL;
  static std::mutex mutex;
  static std::unordered_map<uint64_t, std::weak_ptr<const FibreGruBank>> cache;
  std::lock_guard<std::mutex> lock(mutex);
  if (const auto found = cache.find(key); found != cache.end()) {
    if (auto bank = found->second.lock()) {
      FibreGruPackedBankCacheHits().fetch_add(1, std::memory_order_relaxed);
      return bank;
    }
  }
  auto bank = std::make_shared<FibreGruBank>();
  bank->hidden = hidden;
#if SETRAIN_ARM_FIBRE_GRU_USE_ACL
  if (hidden == FibreGruAclAffine::kHidden) {
    bank->acl = std::make_shared<FibreGruAclAffine>(
        input_weight, recurrent_weight);
  } else {
#elif SETRAIN_ARM_FIBRE_GRU_USE_KLEIDIAI
  if (hidden == 20) {
    bank->kai_h20 = std::make_shared<FibreGruKaiAffine<20, 16>>(
        input_weight, recurrent_weight);
  } else if (hidden == 36) {
    bank->kai_h36 = std::make_shared<FibreGruKaiAffine<36, 24>>(
        input_weight, recurrent_weight);
  } else if (hidden == 48) {
    bank->kai_h48 = std::make_shared<FibreGruKaiAffine<48, 36>>(
        input_weight, recurrent_weight);
  } else if (hidden == 80) {
    bank->kai_h80 = std::make_shared<FibreGruKaiAffine<80, 4>>(
        input_weight, recurrent_weight);
  } else if (hidden == 144) {
    bank->kai_h144 = std::make_shared<FibreGruKaiAffine<144, 4>>(
        input_weight, recurrent_weight);
  } else if (hidden == 160 && batch_hint == 4) {
    bank->kai_h160_b4 = std::make_shared<FibreGruKaiAffine<160, 4>>(
        input_weight, recurrent_weight);
  } else if (hidden == 160 && batch_hint == 8) {
    bank->kai_h160_b8 = std::make_shared<FibreGruKaiAffine<160, 8>>(
        input_weight, recurrent_weight);
  } else if (hidden == 176) {
    bank->kai_h176 = std::make_shared<FibreGruKaiAffine<176, 6>>(
        input_weight, recurrent_weight);
  } else {
#endif
  bank->input.resize(matrix_elements);
  bank->recurrent.resize(matrix_elements);
  constexpr int64_t kOutputTile = 8;
  for (int64_t output = 0; output < 3 * hidden; output += kOutputTile) {
    for (int64_t input = 0; input < hidden; ++input) {
      for (int64_t lane = 0; lane < kOutputTile; ++lane) {
        const size_t packed = static_cast<size_t>(
            output * hidden + input * kOutputTile + lane);
        const size_t source = static_cast<size_t>((output + lane) * hidden + input);
        bank->input[packed] = input_weight[source];
        bank->recurrent[packed] = recurrent_weight[source];
      }
    }
  }
#if SETRAIN_ARM_FIBRE_GRU_USE_ACL || SETRAIN_ARM_FIBRE_GRU_USE_KLEIDIAI
  }
#endif
  cache[key] = bank;
  FibreGruPackedBankCount().fetch_add(1, std::memory_order_relaxed);
  return bank;
}

inline float Logistic(float value) {
  return 1.0F / (1.0F + std::exp(-value));
}

#if defined(__aarch64__)
inline float32x4_t FibreGruExp(float32x4_t value) {
  const float32x4_t log2e = vdupq_n_f32(1.4426950408889634F);
  const float32x4_t ln2 = vdupq_n_f32(0.6931471805599453F);
  value = vmaxq_f32(vdupq_n_f32(-18.0F),
                    vminq_f32(vdupq_n_f32(18.0F), value));
  const float32x4_t rounded = vrndnq_f32(vmulq_f32(value, log2e));
  const int32x4_t exponent = vcvtq_s32_f32(rounded);
  const float32x4_t remainder = vfmsq_f32(value, rounded, ln2);
  float32x4_t polynomial = vdupq_n_f32(1.0F / 720.0F);
  polynomial = vaddq_f32(
      vdupq_n_f32(1.0F / 120.0F), vmulq_f32(polynomial, remainder));
  polynomial = vaddq_f32(
      vdupq_n_f32(1.0F / 24.0F), vmulq_f32(polynomial, remainder));
  polynomial = vaddq_f32(
      vdupq_n_f32(1.0F / 6.0F), vmulq_f32(polynomial, remainder));
  polynomial = vaddq_f32(
      vdupq_n_f32(0.5F), vmulq_f32(polynomial, remainder));
  polynomial = vaddq_f32(
      vdupq_n_f32(1.0F), vmulq_f32(polynomial, remainder));
  polynomial = vaddq_f32(
      vdupq_n_f32(1.0F), vmulq_f32(polynomial, remainder));
  const int32x4_t exponent_bits = vshlq_n_s32(
      vaddq_s32(exponent, vdupq_n_s32(127)), 23);
  return vmulq_f32(polynomial, vreinterpretq_f32_s32(exponent_bits));
}

inline float32x4_t FibreGruLogistic(float32x4_t value) {
  const float32x4_t denominator =
      vaddq_f32(vdupq_n_f32(1.0F), FibreGruExp(vnegq_f32(value)));
  float32x4_t reciprocal = vrecpeq_f32(denominator);
  reciprocal = vmulq_f32(reciprocal, vrecpsq_f32(denominator, reciprocal));
  reciprocal = vmulq_f32(reciprocal, vrecpsq_f32(denominator, reciprocal));
  return reciprocal;
}

inline float32x4_t FibreGruTanh(float32x4_t value) {
  return vsubq_f32(
      vmulq_n_f32(FibreGruLogistic(vmulq_n_f32(value, 2.0F)), 2.0F),
      vdupq_n_f32(1.0F));
}
#endif

template <int Hidden, int Batch, int Outputs = 3 * Hidden>
void FibreGruAffine(
    const float* values, const float* packed_weight,
    const float* bias, float* output) {
  constexpr int kOutputTile = 8;
  constexpr int kOutputs = Outputs;
  for (int output_begin = 0; output_begin < kOutputs;
       output_begin += kOutputTile) {
#if defined(__aarch64__)
    float32x4_t sum0[Batch];
    float32x4_t sum1[Batch];
    for (int item = 0; item < Batch; ++item) {
      sum0[item] = vld1q_f32(bias + output_begin);
      sum1[item] = vld1q_f32(bias + output_begin + 4);
    }
    const float* weight = packed_weight + output_begin * Hidden;
    for (int input = 0; input < Hidden; ++input) {
      const float32x4_t weight0 = vld1q_f32(weight + input * kOutputTile);
      const float32x4_t weight1 = vld1q_f32(weight + input * kOutputTile + 4);
      for (int item = 0; item < Batch; ++item) {
        const float scalar = values[item * Hidden + input];
        sum0[item] = vfmaq_n_f32(sum0[item], weight0, scalar);
        sum1[item] = vfmaq_n_f32(sum1[item], weight1, scalar);
      }
    }
    for (int item = 0; item < Batch; ++item) {
      vst1q_f32(output + item * kOutputs + output_begin, sum0[item]);
      vst1q_f32(output + item * kOutputs + output_begin + 4, sum1[item]);
    }
#else
    const float* weight = packed_weight + output_begin * Hidden;
    for (int item = 0; item < Batch; ++item) {
      for (int lane = 0; lane < kOutputTile; ++lane) {
        float sum = bias[output_begin + lane];
        for (int input = 0; input < Hidden; ++input) {
          sum = std::fma(
              values[item * Hidden + input],
              weight[input * kOutputTile + lane], sum);
        }
        output[item * kOutputs + output_begin + lane] = sum;
      }
    }
#endif
  }
}

template <int Hidden, int Batch>
void FibreGruDualAffine(
    const float* first_values, const float* second_values,
    const float* first_weight, const float* second_weight,
    const float* first_bias, const float* second_bias,
    float* first_output, float* second_output) {
  constexpr int kOutputTile = 8;
  for (int output_begin = 0; output_begin < Hidden;
       output_begin += kOutputTile) {
#if defined(__aarch64__)
    float32x4_t first0[Batch];
    float32x4_t first1[Batch];
    float32x4_t second0[Batch];
    float32x4_t second1[Batch];
    for (int item = 0; item < Batch; ++item) {
      first0[item] = vld1q_f32(first_bias + output_begin);
      first1[item] = vld1q_f32(first_bias + output_begin + 4);
      second0[item] = vld1q_f32(second_bias + output_begin);
      second1[item] = vld1q_f32(second_bias + output_begin + 4);
    }
    const float* first = first_weight + output_begin * Hidden;
    const float* second = second_weight + output_begin * Hidden;
    for (int input = 0; input < Hidden; ++input) {
      const float32x4_t first_weight0 =
          vld1q_f32(first + input * kOutputTile);
      const float32x4_t first_weight1 =
          vld1q_f32(first + input * kOutputTile + 4);
      const float32x4_t second_weight0 =
          vld1q_f32(second + input * kOutputTile);
      const float32x4_t second_weight1 =
          vld1q_f32(second + input * kOutputTile + 4);
      for (int item = 0; item < Batch; ++item) {
        const float first_scalar = first_values[item * Hidden + input];
        const float second_scalar = second_values[item * Hidden + input];
        first0[item] =
            vfmaq_n_f32(first0[item], first_weight0, first_scalar);
        first1[item] =
            vfmaq_n_f32(first1[item], first_weight1, first_scalar);
        second0[item] =
            vfmaq_n_f32(second0[item], second_weight0, second_scalar);
        second1[item] =
            vfmaq_n_f32(second1[item], second_weight1, second_scalar);
      }
    }
    for (int item = 0; item < Batch; ++item) {
      vst1q_f32(first_output + item * Hidden + output_begin, first0[item]);
      vst1q_f32(first_output + item * Hidden + output_begin + 4, first1[item]);
      vst1q_f32(second_output + item * Hidden + output_begin, second0[item]);
      vst1q_f32(second_output + item * Hidden + output_begin + 4, second1[item]);
    }
#else
    for (int item = 0; item < Batch; ++item) {
      for (int lane = 0; lane < kOutputTile; ++lane) {
        float first_sum = first_bias[output_begin + lane];
        float second_sum = second_bias[output_begin + lane];
        for (int input = 0; input < Hidden; ++input) {
          first_sum = std::fma(
              first_values[item * Hidden + input],
              first_weight[
                  output_begin * Hidden + input * kOutputTile + lane],
              first_sum);
          second_sum = std::fma(
              second_values[item * Hidden + input],
              second_weight[
                  output_begin * Hidden + input * kOutputTile + lane],
              second_sum);
        }
        first_output[item * Hidden + output_begin + lane] = first_sum;
        second_output[item * Hidden + output_begin + lane] = second_sum;
      }
    }
#endif
  }
}

template <int Hidden, int Batch>
void FibreGruCombinedGate(
    const float* input, const float* recurrent,
    const float* packed_input_weight, const float* packed_recurrent_weight,
    const float* input_bias, const float* recurrent_bias, float* output) {
  constexpr int kOutputTile = 8;
  for (int output_begin = 0; output_begin < Hidden;
       output_begin += kOutputTile) {
#if defined(__aarch64__)
    float32x4_t sum0[Batch];
    float32x4_t sum1[Batch];
    for (int item = 0; item < Batch; ++item) {
      sum0[item] = vaddq_f32(
          vld1q_f32(input_bias + output_begin),
          vld1q_f32(recurrent_bias + output_begin));
      sum1[item] = vaddq_f32(
          vld1q_f32(input_bias + output_begin + 4),
          vld1q_f32(recurrent_bias + output_begin + 4));
    }
    const float* input_weight = packed_input_weight + output_begin * Hidden;
    const float* recurrent_weight =
        packed_recurrent_weight + output_begin * Hidden;
    for (int inner = 0; inner < Hidden; ++inner) {
      const float32x4_t input_weight0 =
          vld1q_f32(input_weight + inner * kOutputTile);
      const float32x4_t input_weight1 =
          vld1q_f32(input_weight + inner * kOutputTile + 4);
      const float32x4_t recurrent_weight0 =
          vld1q_f32(recurrent_weight + inner * kOutputTile);
      const float32x4_t recurrent_weight1 =
          vld1q_f32(recurrent_weight + inner * kOutputTile + 4);
      for (int item = 0; item < Batch; ++item) {
        const float input_scalar = input[item * Hidden + inner];
        const float recurrent_scalar = recurrent[item * Hidden + inner];
        sum0[item] = vfmaq_n_f32(sum0[item], input_weight0, input_scalar);
        sum0[item] =
            vfmaq_n_f32(sum0[item], recurrent_weight0, recurrent_scalar);
        sum1[item] = vfmaq_n_f32(sum1[item], input_weight1, input_scalar);
        sum1[item] =
            vfmaq_n_f32(sum1[item], recurrent_weight1, recurrent_scalar);
      }
    }
    for (int item = 0; item < Batch; ++item) {
      vst1q_f32(output + item * Hidden + output_begin, sum0[item]);
      vst1q_f32(output + item * Hidden + output_begin + 4, sum1[item]);
    }
#else
    for (int item = 0; item < Batch; ++item) {
      for (int lane = 0; lane < kOutputTile; ++lane) {
        float sum = input_bias[output_begin + lane] +
                    recurrent_bias[output_begin + lane];
        for (int inner = 0; inner < Hidden; ++inner) {
          sum = std::fma(
              input[item * Hidden + inner],
              packed_input_weight[
                  output_begin * Hidden + inner * kOutputTile + lane],
              sum);
          sum = std::fma(
              recurrent[item * Hidden + inner],
              packed_recurrent_weight[
                  output_begin * Hidden + inner * kOutputTile + lane],
              sum);
        }
        output[item * Hidden + output_begin + lane] = sum;
      }
    }
#endif
  }
}

template <int Hidden, int Batch>
void FibreGruPairedGate(
    const float* input, const float* recurrent,
    const float* packed_input_weight, const float* packed_recurrent_weight,
    const float* input_bias, const float* recurrent_bias,
    float* update_output, float* reset_output) {
  alignas(64) float input_update[Batch * Hidden];
  alignas(64) float recurrent_update[Batch * Hidden];
  alignas(64) float input_reset[Batch * Hidden];
  alignas(64) float recurrent_reset[Batch * Hidden];
  FibreGruDualAffine<Hidden, Batch>(
      input, recurrent, packed_input_weight, packed_recurrent_weight,
      input_bias, recurrent_bias, input_update, recurrent_update);
  FibreGruDualAffine<Hidden, Batch>(
      input, recurrent, packed_input_weight + Hidden * Hidden,
      packed_recurrent_weight + Hidden * Hidden, input_bias + Hidden,
      recurrent_bias + Hidden, input_reset, recurrent_reset);
  for (int index = 0; index < Batch * Hidden; ++index) {
    update_output[index] = input_update[index] + recurrent_update[index];
    reset_output[index] = input_reset[index] + recurrent_reset[index];
  }
}

template <int Hidden, int Batch>
void FibreGruFixed(
    const FibreGruBank& bank, const float* bias, const float* input,
    const float* initial, float* sequence, float* final) {
  constexpr int kGates = 3 * Hidden;
  alignas(64) float update_affine[Batch * Hidden];
  alignas(64) float reset_affine[Batch * Hidden];
  alignas(64) float candidate_input[Batch * Hidden];
  alignas(64) float candidate_recurrent[Batch * Hidden];
  FibreGruPairedGate<Hidden, Batch>(
      input, initial, bank.input.data(), bank.recurrent.data(), bias,
      bias + kGates, update_affine, reset_affine);
  FibreGruDualAffine<Hidden, Batch>(
      input, initial, bank.input.data() + 2 * Hidden * Hidden,
      bank.recurrent.data() + 2 * Hidden * Hidden, bias + 2 * Hidden,
      bias + kGates + 2 * Hidden, candidate_input, candidate_recurrent);
  for (int item = 0; item < Batch; ++item) {
    const float* update_values = update_affine + item * Hidden;
    const float* reset_values = reset_affine + item * Hidden;
    const float* input_candidate = candidate_input + item * Hidden;
    const float* recurrent_candidate = candidate_recurrent + item * Hidden;
    const float* previous = initial + item * Hidden;
    float* next = final + item * Hidden;
#if defined(__aarch64__)
    int channel = 0;
    for (; channel + 4 <= Hidden; channel += 4) {
      const float32x4_t update =
          FibreGruLogistic(vld1q_f32(update_values + channel));
      const float32x4_t reset =
          FibreGruLogistic(vld1q_f32(reset_values + channel));
      const float32x4_t candidate = FibreGruTanh(vaddq_f32(
          vld1q_f32(input_candidate + channel),
          vmulq_f32(reset, vld1q_f32(recurrent_candidate + channel))));
      const float32x4_t previous_value = vld1q_f32(previous + channel);
      vst1q_f32(next + channel, vfmaq_f32(
          candidate, update, vsubq_f32(previous_value, candidate)));
    }
    for (; channel < Hidden; ++channel) {
#else
    for (int channel = 0; channel < Hidden; ++channel) {
#endif
      const float update = Logistic(update_values[channel]);
      const float reset = Logistic(reset_values[channel]);
      const float candidate = std::tanh(
          input_candidate[channel] + reset * recurrent_candidate[channel]);
      next[channel] =
          (1.0F - update) * candidate + update * previous[channel];
    }
  }
  std::copy_n(final, Batch * Hidden, sequence);
}

#if SETRAIN_ARM_FIBRE_GRU_USE_ACL || SETRAIN_ARM_FIBRE_GRU_USE_KLEIDIAI
template <int Hidden, int Batch>
void FibreGruKaiEpilogue(
    const float* input_affine, const float* recurrent_affine,
    const float* bias, const float* initial, float* sequence, float* final) {
  constexpr int kHidden = Hidden;
  constexpr int kBatch = Batch;
  constexpr int kGates = 3 * kHidden;
  for (int item = 0; item < kBatch; ++item) {
    const int base = item * kGates;
    const float* previous = initial + item * kHidden;
    float* next = final + item * kHidden;
    for (int channel = 0; channel < kHidden; channel += 4) {
      const float32x4_t update = FibreGruLogistic(vaddq_f32(
          vaddq_f32(
              vld1q_f32(input_affine + base + channel),
              vld1q_f32(recurrent_affine + base + channel)),
          vaddq_f32(
              vld1q_f32(bias + channel),
              vld1q_f32(bias + kGates + channel))));
      const float32x4_t reset = FibreGruLogistic(vaddq_f32(
          vaddq_f32(
              vld1q_f32(input_affine + base + kHidden + channel),
              vld1q_f32(recurrent_affine + base + kHidden + channel)),
          vaddq_f32(
              vld1q_f32(bias + kHidden + channel),
              vld1q_f32(bias + kGates + kHidden + channel))));
      const float32x4_t candidate_input = vaddq_f32(
          vld1q_f32(input_affine + base + 2 * kHidden + channel),
          vld1q_f32(bias + 2 * kHidden + channel));
      const float32x4_t candidate_recurrent = vaddq_f32(
          vld1q_f32(recurrent_affine + base + 2 * kHidden + channel),
          vld1q_f32(bias + kGates + 2 * kHidden + channel));
      const float32x4_t candidate = FibreGruTanh(
          vaddq_f32(candidate_input, vmulq_f32(reset, candidate_recurrent)));
      const float32x4_t previous_value = vld1q_f32(previous + channel);
      vst1q_f32(
          next + channel,
          vfmaq_f32(
              candidate, update, vsubq_f32(previous_value, candidate)));
    }
  }
  std::copy_n(final, kBatch * kHidden, sequence);
}
#endif

#if SETRAIN_ARM_FIBRE_GRU_USE_ACL
void FibreGruAclFixed(
    const FibreGruBank& bank, const float* bias, const float* input,
    const float* initial, float* sequence, float* final) {
  if (!bank.acl) throw std::runtime_error("missing H144 ACL affine bank");
  const auto [input_affine, recurrent_affine] = bank.acl->Run(input, initial);
  FibreGruKaiEpilogue<144, 4>(
      input_affine, recurrent_affine, bias, initial, sequence, final);
}
#endif

#if SETRAIN_ARM_FIBRE_GRU_USE_KLEIDIAI
template <int Hidden, int Batch>
void FibreGruKaiFixed(
    const std::shared_ptr<FibreGruKaiAffine<Hidden, Batch>>& affine,
    const float* bias, const float* input,
    const float* initial, float* sequence, float* final) {
  if (!affine) throw std::runtime_error("missing KleidiAI affine bank");
  const auto [input_affine, recurrent_affine] = affine->Run(input, initial);
  FibreGruKaiEpilogue<Hidden, Batch>(
      input_affine, recurrent_affine, bias, initial, sequence, final);
}

template <int Hidden, int Batch>
void FibreGruKaiFixedSplitInput(
    const std::shared_ptr<FibreGruKaiAffine<Hidden, Batch>>& affine,
    const float* bias, const float* const* first_rows, size_t first_columns,
    const float* const* second_rows, size_t second_columns,
    const float* initial, float* sequence, float* final) {
  if (!affine) throw std::runtime_error("missing KleidiAI affine bank");
  const auto [input_affine, recurrent_affine] = affine->RunSplitInput(
      first_rows, first_columns, second_rows, second_columns, initial);
  FibreGruKaiEpilogue<Hidden, Batch>(
      input_affine, recurrent_affine, bias, initial, sequence, final);
}
#endif

struct FibreGru {
  FibreGru(const OrtApi*, const OrtKernelInfo* raw_info) {
    const Ort::ConstKernelInfo info{raw_info};
    int input_constant = 0;
    int recurrent_constant = 0;
    int bias_constant = 0;
    const Ort::ConstValue input_weight =
        info.GetTensorConstantInput(1, &input_constant);
    const Ort::ConstValue recurrent_weight =
        info.GetTensorConstantInput(2, &recurrent_constant);
    const Ort::ConstValue bias = info.GetTensorConstantInput(3, &bias_constant);
    if (!input_constant || !recurrent_constant || !bias_constant) {
      throw std::runtime_error("FibreGRU requires constant W/R/B");
    }
    const auto weight_shape =
        input_weight.GetTensorTypeAndShapeInfo().GetShape();
    const auto recurrent_shape =
        recurrent_weight.GetTensorTypeAndShapeInfo().GetShape();
    hidden_ = weight_shape.size() == 3 ? weight_shape[2] : 0;
    if ((hidden_ != 20 && hidden_ != 36 && hidden_ != 48 && hidden_ != 80 &&
         hidden_ != 144 && hidden_ != 160 && hidden_ != 176) ||
        weight_shape != std::vector<int64_t>({1, 3 * hidden_, hidden_}) ||
        recurrent_shape != weight_shape ||
        bias.GetTensorTypeAndShapeInfo().GetElementCount() !=
            static_cast<size_t>(6 * hidden_)) {
      throw std::runtime_error("unsupported FibreGRU constant shape");
    }
    bank_ = PackFibreGruBank(
        input_weight.GetTensorData<float>(),
        recurrent_weight.GetTensorData<float>(), hidden_);
    bias_.assign(
        bias.GetTensorData<float>(),
        bias.GetTensorData<float>() + 6 * hidden_);
  }

  Ort::Status Compute(
      const Ort::Custom::Tensor<float>& input,
      const Ort::Custom::Tensor<float>&,
      const Ort::Custom::Tensor<float>&,
      const Ort::Custom::Tensor<float>&,
      const Ort::Custom::Tensor<float>& initial,
      Ort::Custom::Tensor<float>& sequence,
      Ort::Custom::Tensor<float>& final) {
    const int64_t elements = input.NumberOfElement();
    if (elements % hidden_ != 0 ||
        initial.NumberOfElement() != elements) {
      return Ort::Status("invalid FibreGRU dynamic input shape", ORT_INVALID_ARGUMENT);
    }
    const int64_t batch = elements / hidden_;
    float* sequence_output = sequence.Allocate({1, 1, batch, hidden_});
    float* final_output = final.Allocate({1, batch, hidden_});
    if (hidden_ == 20 && batch == 16) {
#if SETRAIN_ARM_FIBRE_GRU_USE_KLEIDIAI
      FibreGruKaiFixed<20, 16>(
          bank_->kai_h20, bias_.data(), input.Data(), initial.Data(),
          sequence_output, final_output);
#else
      return Ort::Status("H20/B16 requires KleidiAI", ORT_INVALID_ARGUMENT);
#endif
    } else if (hidden_ == 36 && batch == 24) {
#if SETRAIN_ARM_FIBRE_GRU_USE_KLEIDIAI
      FibreGruKaiFixed<36, 24>(
          bank_->kai_h36, bias_.data(), input.Data(), initial.Data(),
          sequence_output, final_output);
#else
      return Ort::Status("H36/B24 requires KleidiAI", ORT_INVALID_ARGUMENT);
#endif
    } else if (hidden_ == 48 && batch == 36) {
#if SETRAIN_ARM_FIBRE_GRU_USE_KLEIDIAI
      FibreGruKaiFixed<48, 36>(
          bank_->kai_h48, bias_.data(), input.Data(), initial.Data(),
          sequence_output, final_output);
#else
      return Ort::Status("H48/B36 requires KleidiAI", ORT_INVALID_ARGUMENT);
#endif
    } else if (hidden_ == 80 && batch == 4) {
#if SETRAIN_ARM_FIBRE_GRU_USE_KLEIDIAI
      FibreGruKaiFixed<80, 4>(
          bank_->kai_h80, bias_.data(), input.Data(), initial.Data(),
          sequence_output, final_output);
#else
      FibreGruFixed<80, 4>(*bank_, bias_.data(), input.Data(), initial.Data(),
                           sequence_output, final_output);
#endif
    } else if (hidden_ == 144 && batch == 4) {
#if SETRAIN_ARM_FIBRE_GRU_USE_ACL
      FibreGruAclFixed(
          *bank_, bias_.data(), input.Data(), initial.Data(), sequence_output,
          final_output);
#elif SETRAIN_ARM_FIBRE_GRU_USE_KLEIDIAI
      FibreGruKaiFixed<144, 4>(
          bank_->kai_h144, bias_.data(), input.Data(), initial.Data(),
          sequence_output, final_output);
#else
      FibreGruFixed<144, 4>(*bank_, bias_.data(), input.Data(), initial.Data(),
                            sequence_output, final_output);
#endif
    } else if (hidden_ == 160 && batch == 4) {
      FibreGruFixed<160, 4>(*bank_, bias_.data(), input.Data(), initial.Data(),
                            sequence_output, final_output);
    } else if (hidden_ == 176 && batch == 6) {
#if SETRAIN_ARM_FIBRE_GRU_USE_KLEIDIAI
      FibreGruKaiFixed<176, 6>(
          bank_->kai_h176, bias_.data(), input.Data(), initial.Data(),
          sequence_output, final_output);
#else
      FibreGruFixed<176, 6>(*bank_, bias_.data(), input.Data(), initial.Data(),
                            sequence_output, final_output);
#endif
    } else {
      return Ort::Status("unsupported FibreGRU H/batch dispatch", ORT_INVALID_ARGUMENT);
    }
    return Ort::Status{nullptr};
  }

  static Ort::Status InferOutputShape(Ort::ShapeInferContext& context) {
    const auto& input = context.GetInputShape(0);
    if (input.size() != 3 || !input[1].IsInt() || !input[2].IsInt()) {
      return Ort::Status("FibreGRU requires static [1,B,H]", ORT_INVALID_ARGUMENT);
    }
    context.SetOutputShape(0, {{1}, {1}, input[1], input[2]});
    context.SetOutputShape(1, {{1}, input[1], input[2]});
    return Ort::Status{nullptr};
  }

  int64_t hidden_{};
  std::shared_ptr<const FibreGruBank> bank_;
  std::vector<float> bias_;
};

#if SETRAIN_ARM_FIBRE_GRU_USE_KLEIDIAI
struct FibreGruBlock {
  FibreGruBlock(const OrtApi*, const OrtKernelInfo* raw_info) {
    const Ort::ConstKernelInfo info{raw_info};
    auto constant = [&info](size_t index) {
      int is_constant = 0;
      Ort::ConstValue value = info.GetTensorConstantInput(index, &is_constant);
      if (!is_constant) {
        throw std::runtime_error("FibreGRUBlock requires constant inputs");
      }
      return value;
    };
    const Ort::ConstValue input_weight = constant(2);
    const Ort::ConstValue recurrent_weight = constant(3);
    const Ort::ConstValue gru_bias = constant(4);
    const Ort::ConstValue projection_weight = constant(6);
    const Ort::ConstValue projection_bias = constant(7);
    const Ort::ConstValue position = constant(8);
    const Ort::ConstValue group = constant(9);
    const Ort::ConstValue offset = constant(10);
    const auto weight_shape = input_weight.GetTensorTypeAndShapeInfo().GetShape();
    hidden_ = weight_shape.size() == 3 ? weight_shape[2] : 0;
    if ((hidden_ != 80 && hidden_ != 144 && hidden_ != 160 &&
         hidden_ != 176) ||
        weight_shape != std::vector<int64_t>({1, 3 * hidden_, hidden_}) ||
        recurrent_weight.GetTensorTypeAndShapeInfo().GetShape() != weight_shape ||
        gru_bias.GetTensorTypeAndShapeInfo().GetElementCount() !=
            static_cast<size_t>(6 * hidden_) ||
        projection_weight.GetTensorTypeAndShapeInfo().GetShape() !=
            std::vector<int64_t>({hidden_, hidden_}) ||
        projection_bias.GetTensorTypeAndShapeInfo().GetElementCount() !=
            static_cast<size_t>(hidden_) ||
        position.GetTensorTypeAndShapeInfo().GetElementCount() %
            static_cast<size_t>(hidden_) != 0 ||
        group.GetTensorTypeAndShapeInfo().GetElementCount() != 1 ||
        offset.GetTensorTypeAndShapeInfo().GetElementCount() != 1) {
      throw std::runtime_error("unsupported FibreGRUBlock constant shape");
    }
    batch_ = static_cast<int64_t>(
        position.GetTensorTypeAndShapeInfo().GetElementCount()) / hidden_;
    gru_bank_ = PackFibreGruBank(
        input_weight.GetTensorData<float>(),
        recurrent_weight.GetTensorData<float>(), hidden_, batch_);
    gru_bias_.assign(
        gru_bias.GetTensorData<float>(),
        gru_bias.GetTensorData<float>() + 6 * hidden_);
    if (hidden_ == 80) {
      projection_h80_ = std::make_unique<FibreProjectionKai<80, 4>>(
          projection_weight.GetTensorData<float>(),
          projection_bias.GetTensorData<float>());
    } else if (hidden_ == 144) {
      projection_h144_ = std::make_unique<FibreProjectionKai<144, 4>>(
          projection_weight.GetTensorData<float>(),
          projection_bias.GetTensorData<float>());
    } else if (hidden_ == 160 && batch_ == 4) {
      projection_h160_b4_ = std::make_unique<FibreProjectionKai<160, 4>>(
          projection_weight.GetTensorData<float>(),
          projection_bias.GetTensorData<float>());
    } else if (hidden_ == 160 && batch_ == 8) {
      projection_h160_b8_ = std::make_unique<FibreProjectionKai<160, 8>>(
          projection_weight.GetTensorData<float>(),
          projection_bias.GetTensorData<float>());
    } else {
      projection_h176_ = std::make_unique<FibreProjectionKai<176, 6>>(
          projection_weight.GetTensorData<float>(),
          projection_bias.GetTensorData<float>());
    }
    group_ = group.GetTensorData<int64_t>()[0];
    offset_ = offset.GetTensorData<int64_t>()[0];
  }

  Ort::Status Compute(
      const Ort::Custom::Tensor<float>& packed_input,
      const Ort::Custom::Tensor<float>& residual,
      const Ort::Custom::Tensor<float>&,
      const Ort::Custom::Tensor<float>&,
      const Ort::Custom::Tensor<float>&,
      const Ort::Custom::Tensor<float>& initial,
      const Ort::Custom::Tensor<float>&,
      const Ort::Custom::Tensor<float>&,
      const Ort::Custom::Tensor<float>& position,
      const Ort::Custom::Tensor<int64_t>&,
      const Ort::Custom::Tensor<int64_t>&,
      Ort::Custom::Tensor<float>& output,
      Ort::Custom::Tensor<float>& final) {
    if (hidden_ == 80) {
      return ComputeFixed<80, 4>(
          packed_input, residual, initial, position, output, final,
          gru_bank_->kai_h80, projection_h80_.get());
    }
    if (hidden_ == 144) {
      return ComputeFixed<144, 4>(
          packed_input, residual, initial, position, output, final,
          gru_bank_->kai_h144, projection_h144_.get());
    }
    if (hidden_ == 160 && batch_ == 4) {
      return ComputeFixed<160, 4>(
          packed_input, residual, initial, position, output, final,
          gru_bank_->kai_h160_b4, projection_h160_b4_.get());
    }
    if (hidden_ == 160 && batch_ == 8) {
      return ComputeFixed<160, 8>(
          packed_input, residual, initial, position, output, final,
          gru_bank_->kai_h160_b8, projection_h160_b8_.get());
    }
    if (hidden_ == 176) {
      return ComputeFixed<176, 6>(
          packed_input, residual, initial, position, output, final,
          gru_bank_->kai_h176, projection_h176_.get());
    }
    return Ort::Status("unsupported FibreGRUBlock dispatch", ORT_INVALID_ARGUMENT);
  }

  template <int Hidden, int Batch>
  Ort::Status ComputeFixed(
      const Ort::Custom::Tensor<float>& packed_input,
      const Ort::Custom::Tensor<float>& residual,
      const Ort::Custom::Tensor<float>& initial,
      const Ort::Custom::Tensor<float>& position,
      Ort::Custom::Tensor<float>& output,
      Ort::Custom::Tensor<float>& final,
      const std::shared_ptr<FibreGruKaiAffine<Hidden, Batch>>& affine,
      FibreProjectionKai<Hidden, Batch>* projection) {
    if (group_ <= 0 || Hidden % group_ != 0 ||
        packed_input.NumberOfElement() != Batch * Hidden ||
        initial.NumberOfElement() != Batch * Hidden || projection == nullptr) {
      return Ort::Status("invalid FibreGRUBlock input shape", ORT_INVALID_ARGUMENT);
    }
    const auto& residual_shape = residual.Shape();
    if (residual_shape.size() < 2) {
      return Ort::Status("invalid FibreGRUBlock residual", ORT_INVALID_ARGUMENT);
    }
    const int64_t frequencies = residual_shape[residual_shape.size() - 2];
    const int64_t channels = residual_shape.back();
    if (frequencies / group_ != Batch || frequencies % group_ != 0 ||
        channels * group_ != Hidden ||
        residual.NumberOfElement() != frequencies * channels ||
        position.NumberOfElement() != frequencies * channels) {
      return Ort::Status("unsupported FibreGRUBlock layout", ORT_INVALID_ARGUMENT);
    }
    const float* unpacked_input = packed_input.Data();
    alignas(64) float sequence[Batch * Hidden];
    float* final_output = final.Allocate({1, Batch, Hidden});
    if (offset_ == 0) {
      FibreGruKaiFixed<Hidden, Batch>(
          affine, gru_bias_.data(), unpacked_input,
          initial.Data(), sequence, final_output);
    } else {
      if (offset_ >= group_) {
        return Ort::Status(
            "unsupported FibreGRUBlock offset", ORT_INVALID_ARGUMENT);
      }
      const int64_t first_frequencies = group_ - offset_;
      const int64_t second_frequencies = offset_;
      const size_t first_columns =
          static_cast<size_t>(first_frequencies * channels);
      const size_t second_columns =
          static_cast<size_t>(second_frequencies * channels);
      const float* first_rows[Batch];
      const float* second_rows[Batch];
      for (int row = 0; row < Batch; ++row) {
        const int64_t first_frequency = offset_ + row * group_;
        const int64_t second_frequency =
            (first_frequency + first_frequencies) % frequencies;
        first_rows[row] = unpacked_input + first_frequency * channels;
        second_rows[row] = unpacked_input + second_frequency * channels;
      }
      FibreGruKaiFixedSplitInput<Hidden, Batch>(
          affine, gru_bias_.data(), first_rows, first_columns,
          second_rows, second_columns, initial.Data(), sequence, final_output);
    }
    const float* projected = projection->Run(final_output);
    const float* residual_data = residual.Data();
    const float* position_data = position.Data();
    float* destination = output.Allocate(residual_shape);
    for (int64_t frequency = 0; frequency < frequencies; ++frequency) {
      const int64_t packed_frequency =
          (frequency - offset_ + frequencies) % frequencies;
      const float* residual_row = residual_data + frequency * channels;
      const float* projected_row = projected + packed_frequency * channels;
      float* destination_row = destination + frequency * channels;
      int64_t channel = 0;
      for (; channel + 8 <= channels; channel += 8) {
        float32x4_t sum0 = vaddq_f32(
            vaddq_f32(
                vld1q_f32(residual_row + channel),
                vld1q_f32(projected_row + channel)),
            vld1q_f32(position_data + frequency * channels + channel));
        float32x4_t sum1 = vaddq_f32(
            vaddq_f32(
                vld1q_f32(residual_row + channel + 4),
                vld1q_f32(projected_row + channel + 4)),
            vld1q_f32(position_data + frequency * channels + channel + 4));
        vst1q_f32(destination_row + channel, sum0);
        vst1q_f32(destination_row + channel + 4, sum1);
      }
      for (; channel < channels; ++channel) {
        destination_row[channel] =
            residual_row[channel] + projected_row[channel] +
            position_data[frequency * channels + channel];
      }
    }
    return Ort::Status{nullptr};
  }

  static Ort::Status InferOutputShape(Ort::ShapeInferContext& context) {
    context.SetOutputShape(0, context.GetInputShape(1));
    context.SetOutputShape(1, context.GetInputShape(5));
    return Ort::Status{nullptr};
  }

  int64_t hidden_{};
  int64_t batch_{};
  int64_t group_{};
  int64_t offset_{};
  std::shared_ptr<const FibreGruBank> gru_bank_;
  std::vector<float> gru_bias_;
  std::unique_ptr<FibreProjectionKai<80, 4>> projection_h80_;
  std::unique_ptr<FibreProjectionKai<144, 4>> projection_h144_;
  std::unique_ptr<FibreProjectionKai<160, 4>> projection_h160_b4_;
  std::unique_ptr<FibreProjectionKai<160, 8>> projection_h160_b8_;
  std::unique_ptr<FibreProjectionKai<176, 6>> projection_h176_;
};

struct FastEnhancerGruBlock {
  FastEnhancerGruBlock(const OrtApi*, const OrtKernelInfo* raw_info) {
    const Ort::ConstKernelInfo info{raw_info};
    auto constant = [&info](size_t index) {
      int is_constant = 0;
      Ort::ConstValue value = info.GetTensorConstantInput(index, &is_constant);
      if (!is_constant) {
        throw std::runtime_error("FastEnhancerGRUBlock requires constants");
      }
      return value;
    };
    const Ort::ConstValue input_weight = constant(2);
    const Ort::ConstValue recurrent_weight = constant(3);
    const Ort::ConstValue gru_bias = constant(4);
    const Ort::ConstValue projection_weight = constant(6);
    const Ort::ConstValue projection_bias = constant(7);
    const Ort::ConstValue position = constant(8);
    const auto weight_shape = input_weight.GetTensorTypeAndShapeInfo().GetShape();
    hidden_ = weight_shape.size() == 3 ? weight_shape[2] : 0;
    batch_ = hidden_ > 0
        ? static_cast<int64_t>(position.GetTensorTypeAndShapeInfo().GetElementCount()) / hidden_
        : 0;
    const bool supported =
        (hidden_ == 20 && batch_ == 16) ||
        (hidden_ == 36 && batch_ == 24) ||
        (hidden_ == 48 && batch_ == 36);
    if (!supported ||
        weight_shape != std::vector<int64_t>({1, 3 * hidden_, hidden_}) ||
        recurrent_weight.GetTensorTypeAndShapeInfo().GetShape() != weight_shape ||
        gru_bias.GetTensorTypeAndShapeInfo().GetElementCount() !=
            static_cast<size_t>(6 * hidden_) ||
        projection_weight.GetTensorTypeAndShapeInfo().GetShape() !=
            std::vector<int64_t>({hidden_, hidden_}) ||
        projection_bias.GetTensorTypeAndShapeInfo().GetElementCount() !=
            static_cast<size_t>(hidden_)) {
      throw std::runtime_error("unsupported FastEnhancerGRUBlock shape");
    }
    gru_bank_ = PackFibreGruBank(
        input_weight.GetTensorData<float>(),
        recurrent_weight.GetTensorData<float>(), hidden_, batch_);
    gru_bias_.assign(
        gru_bias.GetTensorData<float>(),
        gru_bias.GetTensorData<float>() + 6 * hidden_);
    if (hidden_ == 20) {
      projection_h20_ = std::make_unique<FibreProjectionKai<20, 16>>(
          projection_weight.GetTensorData<float>(),
          projection_bias.GetTensorData<float>());
    } else if (hidden_ == 36) {
      projection_h36_ = std::make_unique<FibreProjectionKai<36, 24>>(
          projection_weight.GetTensorData<float>(),
          projection_bias.GetTensorData<float>());
    } else {
      projection_h48_ = std::make_unique<FibreProjectionKai<48, 36>>(
          projection_weight.GetTensorData<float>(),
          projection_bias.GetTensorData<float>());
    }
  }

  Ort::Status Compute(
      const Ort::Custom::Tensor<float>& input,
      const Ort::Custom::Tensor<float>& residual,
      const Ort::Custom::Tensor<float>&,
      const Ort::Custom::Tensor<float>&,
      const Ort::Custom::Tensor<float>&,
      const Ort::Custom::Tensor<float>& initial,
      const Ort::Custom::Tensor<float>&,
      const Ort::Custom::Tensor<float>&,
      const Ort::Custom::Tensor<float>& position,
      Ort::Custom::Tensor<float>& output,
      Ort::Custom::Tensor<float>& final) {
    if (hidden_ == 20 && batch_ == 16) {
      return ComputeFixed<20, 16>(
          input, residual, initial, position, output, final,
          gru_bank_->kai_h20, projection_h20_.get());
    }
    if (hidden_ == 36 && batch_ == 24) {
      return ComputeFixed<36, 24>(
          input, residual, initial, position, output, final,
          gru_bank_->kai_h36, projection_h36_.get());
    }
    if (hidden_ == 48 && batch_ == 36) {
      return ComputeFixed<48, 36>(
          input, residual, initial, position, output, final,
          gru_bank_->kai_h48, projection_h48_.get());
    }
    return Ort::Status("unsupported FastEnhancerGRUBlock dispatch", ORT_INVALID_ARGUMENT);
  }

  template <int Hidden, int Batch>
  Ort::Status ComputeFixed(
      const Ort::Custom::Tensor<float>& input,
      const Ort::Custom::Tensor<float>& residual,
      const Ort::Custom::Tensor<float>& initial,
      const Ort::Custom::Tensor<float>& position,
      Ort::Custom::Tensor<float>& output,
      Ort::Custom::Tensor<float>& final,
      const std::shared_ptr<FibreGruKaiAffine<Hidden, Batch>>& affine,
      FibreProjectionKai<Hidden, Batch>* projection) {
    constexpr int kElements = Hidden * Batch;
    if (input.NumberOfElement() != kElements ||
        residual.NumberOfElement() != kElements ||
        initial.NumberOfElement() != kElements ||
        position.NumberOfElement() != kElements || projection == nullptr) {
      return Ort::Status(
          "invalid FastEnhancerGRUBlock input", ORT_INVALID_ARGUMENT);
    }
    alignas(64) float sequence[kElements];
    float* final_output = final.Allocate({1, Batch, Hidden});
    FibreGruKaiFixed<Hidden, Batch>(
        affine, gru_bias_.data(), input.Data(), initial.Data(), sequence,
        final_output);
    const float* projected = projection->Run(final_output);
    const float* residual_data = residual.Data();
    const float* position_data = position.Data();
    float* destination = output.Allocate(residual.Shape());
    int index = 0;
    for (; index + 8 <= kElements; index += 8) {
      vst1q_f32(
          destination + index,
          vaddq_f32(
              vaddq_f32(vld1q_f32(projected + index),
                        vld1q_f32(residual_data + index)),
              vld1q_f32(position_data + index)));
      vst1q_f32(
          destination + index + 4,
          vaddq_f32(
              vaddq_f32(vld1q_f32(projected + index + 4),
                        vld1q_f32(residual_data + index + 4)),
              vld1q_f32(position_data + index + 4)));
    }
    for (; index < kElements; ++index) {
      destination[index] =
          projected[index] + residual_data[index] + position_data[index];
    }
    return Ort::Status{nullptr};
  }

  static Ort::Status InferOutputShape(Ort::ShapeInferContext& context) {
    context.SetOutputShape(0, context.GetInputShape(1));
    context.SetOutputShape(1, context.GetInputShape(5));
    return Ort::Status{nullptr};
  }

  int64_t hidden_{};
  int64_t batch_{};
  std::shared_ptr<const FibreGruBank> gru_bank_;
  std::vector<float> gru_bias_;
  std::unique_ptr<FibreProjectionKai<20, 16>> projection_h20_;
  std::unique_ptr<FibreProjectionKai<36, 24>> projection_h36_;
  std::unique_ptr<FibreProjectionKai<48, 36>> projection_h48_;
};

#endif

Ort::Status CompressComplexSpectrum(
    const Ort::Custom::Tensor<float>& spectrum,
    Ort::Custom::Tensor<float>& features,
    Ort::Custom::Tensor<float>& compressed) {
  constexpr float kMinimumPower = 1.0e-10F;
  constexpr float kExponent = -0.35F;
  if (spectrum.NumberOfElement() % (kInputFrequencies * 2) != 0) {
    return Ort::Status("invalid complex spectrum", ORT_INVALID_ARGUMENT);
  }
  const int64_t batch = spectrum.NumberOfElement() / (kInputFrequencies * 2);
  const float* input = spectrum.Data();
  float* feature_output = features.Allocate({batch, 2, kBodyFrequencies});
  float* compressed_output = compressed.Allocate({batch, kBodyFrequencies, 1, 2});
  for (int64_t item = 0; item < batch; ++item) {
    const float* source = input + item * kInputFrequencies * 2;
    float* item_features = feature_output + item * kBodyFrequencies * 2;
    float* item_compressed = compressed_output + item * kBodyFrequencies * 2;
    for (int64_t frequency = 0; frequency < kBodyFrequencies; ++frequency) {
      const float real = source[2 * frequency];
      const float imaginary = source[2 * frequency + 1];
      const float power = std::max(real * real + imaginary * imaginary, kMinimumPower);
      const float scale = std::pow(power, kExponent);
      item_compressed[2 * frequency] = real * scale;
      item_compressed[2 * frequency + 1] = imaginary * scale;
      item_features[frequency] = real * scale;
      item_features[kBodyFrequencies + frequency] = imaginary * scale;
    }
  }
  return Ort::Status{nullptr};
}

Ort::Status CompressComplexSpectrumShape(Ort::ShapeInferContext& context) {
  Ort::ShapeInferContext::Shape features = {{1}, {2}, {256}};
  Ort::ShapeInferContext::Shape compressed = {{1}, {256}, {1}, {2}};
  context.SetOutputShape(0, features);
  context.SetOutputShape(1, compressed);
  return Ort::Status{nullptr};
}

Ort::Status CompressComplexSpectrumAndMagnitude(
    const Ort::Custom::Tensor<float>& spectrum,
    Ort::Custom::Tensor<float>& features,
    Ort::Custom::Tensor<float>& compressed) {
  constexpr float kMinimumPower = 1.0e-10F;
  constexpr float kExponent = -0.35F;
  if (spectrum.NumberOfElement() % (kInputFrequencies * 2) != 0) {
    return Ort::Status("invalid complex spectrum", ORT_INVALID_ARGUMENT);
  }
  const int64_t batch = spectrum.NumberOfElement() / (kInputFrequencies * 2);
  const float* input = spectrum.Data();
  float* feature_output = features.Allocate({batch, 3, kBodyFrequencies});
  float* compressed_output = compressed.Allocate({batch, kBodyFrequencies, 1, 2});
  for (int64_t item = 0; item < batch; ++item) {
    const float* source = input + item * kInputFrequencies * 2;
    float* item_features = feature_output + item * kBodyFrequencies * 3;
    float* feature_real = item_features;
    float* feature_imaginary = feature_real + kBodyFrequencies;
    float* feature_magnitude = feature_imaginary + kBodyFrequencies;
    float* item_compressed = compressed_output + item * kBodyFrequencies * 2;
    for (int64_t frequency = 0; frequency < kBodyFrequencies; ++frequency) {
      const float real = source[2 * frequency];
      const float imaginary = source[2 * frequency + 1];
      const float power = std::max(
          real * real + imaginary * imaginary, kMinimumPower);
      const float scale = std::pow(power, kExponent);
      const float compressed_real = real * scale;
      const float compressed_imaginary = imaginary * scale;
      item_compressed[2 * frequency] = compressed_real;
      item_compressed[2 * frequency + 1] = compressed_imaginary;
      feature_real[frequency] = compressed_real;
      feature_imaginary[frequency] = compressed_imaginary;
      feature_magnitude[frequency] = std::sqrt(std::max(
          compressed_real * compressed_real +
              compressed_imaginary * compressed_imaginary,
          kMinimumPower));
    }
  }
  return Ort::Status{nullptr};
}

Ort::Status CompressComplexSpectrumAndMagnitudeShape(
    Ort::ShapeInferContext& context) {
  Ort::ShapeInferContext::Shape features = {{1}, {3}, {256}};
  Ort::ShapeInferContext::Shape compressed = {{1}, {256}, {1}, {2}};
  context.SetOutputShape(0, features);
  context.SetOutputShape(1, compressed);
  return Ort::Status{nullptr};
}

Ort::Status CompressComplexSpectrumAndOrientedGram(
    const Ort::Custom::Tensor<float>& spectrum,
    Ort::Custom::Tensor<float>& observation,
    Ort::Custom::Tensor<float>& compressed) {
  constexpr float kMinimumPower = 1.0e-10F;
  constexpr float kExponent = -0.35F;
  if (spectrum.NumberOfElement() % (kInputFrequencies * 2) != 0) {
    return Ort::Status("invalid complex spectrum", ORT_INVALID_ARGUMENT);
  }
  const int64_t batch = spectrum.NumberOfElement() / (kInputFrequencies * 2);
  const float* input = spectrum.Data();
  float* gram_output = observation.Allocate({batch, 3, kBodyFrequencies});
  float* compressed_output = compressed.Allocate({batch, kBodyFrequencies, 1, 2});
  for (int64_t item = 0; item < batch; ++item) {
    const float* source = input + item * kInputFrequencies * 2;
    float* item_gram = gram_output + item * kBodyFrequencies * 3;
    float* node = item_gram;
    float* edge_real = node + kBodyFrequencies;
    float* edge_imaginary = edge_real + kBodyFrequencies;
    float* item_compressed = compressed_output + item * kBodyFrequencies * 2;
#if SETRAIN_ARM_REUSE_GRAM_POW
    float previous_real = 0.0F;
    float previous_imaginary = 0.0F;
#endif
    for (int64_t frequency = 0; frequency < kBodyFrequencies; ++frequency) {
      const float real = source[2 * frequency];
      const float imaginary = source[2 * frequency + 1];
      const float power = std::max(real * real + imaginary * imaginary, kMinimumPower);
      const float scale = std::pow(power, kExponent);
      const float compressed_real = real * scale;
      const float compressed_imaginary = imaginary * scale;
      item_compressed[2 * frequency] = compressed_real;
      item_compressed[2 * frequency + 1] = compressed_imaginary;
      node[frequency] = compressed_real * compressed_real + compressed_imaginary * compressed_imaginary;
#if SETRAIN_ARM_REUSE_GRAM_POW
      if (frequency != 0) {
        edge_real[frequency - 1] =
            compressed_real * previous_real +
            compressed_imaginary * previous_imaginary;
        edge_imaginary[frequency - 1] =
            compressed_imaginary * previous_real -
            compressed_real * previous_imaginary;
      }
      previous_real = compressed_real;
      previous_imaginary = compressed_imaginary;
#else
      if (frequency + 1 < kBodyFrequencies) {
        const float right_real = source[2 * (frequency + 1)];
        const float right_imaginary = source[2 * (frequency + 1) + 1];
        const float right_power = std::max(
            right_real * right_real + right_imaginary * right_imaginary,
            kMinimumPower);
        const float right_scale = std::pow(right_power, kExponent);
        const float rr = right_real * right_scale;
        const float ri = right_imaginary * right_scale;
        edge_real[frequency] = rr * compressed_real + ri * compressed_imaginary;
        edge_imaginary[frequency] =
            ri * compressed_real - rr * compressed_imaginary;
      } else {
        edge_real[frequency] = 0.0F;
        edge_imaginary[frequency] = 0.0F;
      }
#endif
    }
#if SETRAIN_ARM_REUSE_GRAM_POW
    edge_real[kBodyFrequencies - 1] = 0.0F;
    edge_imaginary[kBodyFrequencies - 1] = 0.0F;
#endif
  }
  return Ort::Status{nullptr};
}

Ort::Status CompressComplexSpectrumAndOrientedGramShape(
    Ort::ShapeInferContext& context) {
  Ort::ShapeInferContext::Shape observation = {{1}, {3}, {256}};
  Ort::ShapeInferContext::Shape compressed = {{1}, {256}, {1}, {2}};
  context.SetOutputShape(0, observation);
  context.SetOutputShape(1, compressed);
  return Ort::Status{nullptr};
}

Ort::Status CompressComplexSpectrumAndOrientedGramConv(
    const Ort::Custom::Tensor<float>& spectrum,
    const Ort::Custom::Tensor<float>& weight,
    const Ort::Custom::Tensor<float>& bias,
    Ort::Custom::Tensor<float>& features,
    Ort::Custom::Tensor<float>& compressed) {
  constexpr float kMinimumPower = 1.0e-10F;
  constexpr float kExponent = -0.35F;
  constexpr int64_t kPaddedFrequencies = 264;
  if (spectrum.NumberOfElement() % (kInputFrequencies * 2) != 0 ||
      bias.NumberOfElement() <= 0 ||
      weight.NumberOfElement() != bias.NumberOfElement() *
                                      kObservationChannels *
                                      kObservationKernelSize) {
    const std::string message =
        "invalid fused Gram convolution interface: spectrum=" +
        std::to_string(spectrum.NumberOfElement()) + ", weight=" +
        std::to_string(weight.NumberOfElement()) + ", bias=" +
        std::to_string(bias.NumberOfElement());
    return Ort::Status(message.c_str(), ORT_INVALID_ARGUMENT);
  }
  const int64_t batch = spectrum.NumberOfElement() / (kInputFrequencies * 2);
  const int64_t output_channels = bias.NumberOfElement();
  const float* input = spectrum.Data();
  const float* weight_data = weight.Data();
  const float* bias_data = bias.Data();
  float* feature_output = features.Allocate(
      {batch, output_channels, kObservationOutputFrequencies});
  float* compressed_output = compressed.Allocate(
      {batch, kBodyFrequencies, 1, 2});
  alignas(64) float observation[kObservationChannels * kPaddedFrequencies];

  for (int64_t item = 0; item < batch; ++item) {
    std::fill_n(observation, kObservationChannels * kPaddedFrequencies, 0.0F);
    float* node = observation + kObservationPadding;
    float* edge_real = observation + kPaddedFrequencies + kObservationPadding;
    float* edge_imaginary =
        observation + 2 * kPaddedFrequencies + kObservationPadding;
    const float* source = input + item * kInputFrequencies * 2;
    float* item_compressed = compressed_output + item * kBodyFrequencies * 2;
    float previous_real = 0.0F;
    float previous_imaginary = 0.0F;
    for (int64_t frequency = 0; frequency < kBodyFrequencies; ++frequency) {
      const float real = source[2 * frequency];
      const float imaginary = source[2 * frequency + 1];
      const float power = std::max(
          real * real + imaginary * imaginary, kMinimumPower);
      const float scale = std::pow(power, kExponent);
      const float compressed_real = real * scale;
      const float compressed_imaginary = imaginary * scale;
      item_compressed[2 * frequency] = compressed_real;
      item_compressed[2 * frequency + 1] = compressed_imaginary;
      node[frequency] =
          compressed_real * compressed_real +
          compressed_imaginary * compressed_imaginary;
      if (frequency != 0) {
        edge_real[frequency - 1] =
            compressed_real * previous_real +
            compressed_imaginary * previous_imaginary;
        edge_imaginary[frequency - 1] =
            compressed_imaginary * previous_real -
            compressed_real * previous_imaginary;
      }
      previous_real = compressed_real;
      previous_imaginary = compressed_imaginary;
    }

    float* item_features = feature_output +
        item * output_channels * kObservationOutputFrequencies;
    for (int64_t output_channel = 0;
         output_channel < output_channels; ++output_channel) {
      const float* channel_weight = weight_data +
          output_channel * kObservationChannels * kObservationKernelSize;
      float* channel_output =
          item_features + output_channel * kObservationOutputFrequencies;
      int64_t output_frequency = 0;
#if defined(__aarch64__)
      for (; output_frequency + 4 <= kObservationOutputFrequencies;
           output_frequency += 4) {
        float32x4_t values = vdupq_n_f32(bias_data[output_channel]);
        for (int64_t observation_channel = 0;
             observation_channel < kObservationChannels;
             ++observation_channel) {
          const float* channel_observation =
              observation + observation_channel * kPaddedFrequencies;
          const float* kernel = channel_weight +
              observation_channel * kObservationKernelSize;
          const int64_t input_begin = output_frequency * kObservationStride;
          for (int64_t tap = 0; tap < kObservationKernelSize; ++tap) {
            const float32x4x4_t gathered =
                vld4q_f32(channel_observation + input_begin + tap);
            values = vfmaq_n_f32(values, gathered.val[0], kernel[tap]);
          }
        }
        vst1q_f32(channel_output + output_frequency, values);
      }
#endif
      for (; output_frequency < kObservationOutputFrequencies;
           ++output_frequency) {
        float value = bias_data[output_channel];
        const int64_t input_begin = output_frequency * kObservationStride;
        for (int64_t observation_channel = 0;
             observation_channel < kObservationChannels;
             ++observation_channel) {
          const float* channel_observation =
              observation + observation_channel * kPaddedFrequencies;
          const float* kernel = channel_weight +
              observation_channel * kObservationKernelSize;
          for (int64_t tap = 0; tap < kObservationKernelSize; ++tap) {
            value = std::fma(
                channel_observation[input_begin + tap], kernel[tap], value);
          }
        }
        channel_output[output_frequency] = value;
      }
    }
  }
  return Ort::Status{nullptr};
}

Ort::Status CompressComplexSpectrumAndOrientedGramConvShape(
    Ort::ShapeInferContext& context) {
  const auto& weight = context.GetInputShape(1);
  if (weight.size() != 3 || !weight[0].IsInt()) {
    return Ort::Status("fused Gram convolution requires static [C,3,9] weight", ORT_INVALID_ARGUMENT);
  }
  Ort::ShapeInferContext::Shape features = {{1}, weight[0], {64}};
  Ort::ShapeInferContext::Shape compressed = {{1}, {256}, {1}, {2}};
  context.SetOutputShape(0, features);
  context.SetOutputShape(1, compressed);
  return Ort::Status{nullptr};
}

Ort::Status ApplyMaskAndDecompress(
    const Ort::Custom::Tensor<float>& compressed,
    const Ort::Custom::Tensor<float>& mask,
    Ort::Custom::Tensor<float>& spectrum) {
  constexpr float kMinimumPower = 1.0e-10F;
  constexpr float kExponent = 1.1666666269302368F;
  if (compressed.NumberOfElement() % (kBodyFrequencies * 2) != 0) {
    return Ort::Status("invalid compressed spectrum", ORT_INVALID_ARGUMENT);
  }
  const int64_t batch = compressed.NumberOfElement() / (kBodyFrequencies * 2);
  if (mask.NumberOfElement() != batch * kBodyFrequencies * 2) {
    return Ort::Status("invalid complex mask", ORT_INVALID_ARGUMENT);
  }
  const float* body = compressed.Data();
  const float* mask_data = mask.Data();
  float* output = spectrum.Allocate({batch, kInputFrequencies, 1, 2});
  for (int64_t item = 0; item < batch; ++item) {
    const float* item_body = body + item * kBodyFrequencies * 2;
    const float* item_mask = mask_data + item * kBodyFrequencies * 2;
    float* item_output = output + item * kInputFrequencies * 2;
    for (int64_t frequency = 0; frequency < kBodyFrequencies; ++frequency) {
      const float real = item_body[2 * frequency];
      const float imaginary = item_body[2 * frequency + 1];
      const float mask_real = item_mask[frequency];
      const float mask_imaginary = item_mask[kBodyFrequencies + frequency];
      const float enhanced_real = real * mask_real - imaginary * mask_imaginary;
      const float enhanced_imaginary = real * mask_imaginary + imaginary * mask_real;
      const float power = std::max(
          enhanced_real * enhanced_real + enhanced_imaginary * enhanced_imaginary,
          kMinimumPower);
      const float scale = std::pow(power, kExponent);
      item_output[2 * frequency] = enhanced_real * scale;
      item_output[2 * frequency + 1] = enhanced_imaginary * scale;
    }
    item_output[2 * kBodyFrequencies] = 0.0F;
    item_output[2 * kBodyFrequencies + 1] = 0.0F;
  }
  return Ort::Status{nullptr};
}

Ort::Status ApplyMaskAndDecompressShape(Ort::ShapeInferContext& context) {
  Ort::ShapeInferContext::Shape spectrum = {{1}, {257}, {1}, {2}};
  context.SetOutputShape(0, spectrum);
  return Ort::Status{nullptr};
}

Ort::Status PackFibre(
    const Ort::Custom::Tensor<float>& input,
    const Ort::Custom::Tensor<int64_t>& group_input,
    const Ort::Custom::Tensor<int64_t>& offset_input,
    Ort::Custom::Tensor<float>& output) {
  const auto& shape = input.Shape();
  if (shape.size() < 2 || group_input.NumberOfElement() != 1 ||
      offset_input.NumberOfElement() != 1) {
    return Ort::Status("invalid Fibre pack interface", ORT_INVALID_ARGUMENT);
  }
  const int64_t frequencies = shape[shape.size() - 2];
  const int64_t channels = shape.back();
  const int64_t group = group_input.Data()[0];
  const int64_t offset = offset_input.Data()[0];
  if (frequencies <= 0 || channels <= 0 || group <= 0 ||
      frequencies % group != 0 || offset < 0 || offset >= frequencies ||
      input.NumberOfElement() != frequencies * channels) {
    return Ort::Status("unsupported Fibre pack shape", ORT_INVALID_ARGUMENT);
  }
  const float* source = input.Data();
  float* destination = output.Allocate({1, frequencies / group, group * channels});
#if SETRAIN_ARM_OPTIMIZED_FIBRE_LAYOUT
  const size_t tail_elements =
      static_cast<size_t>((frequencies - offset) * channels);
  std::memcpy(destination, source + offset * channels,
              tail_elements * sizeof(float));
  if (offset != 0) {
    const size_t head_elements = static_cast<size_t>(offset * channels);
    std::memcpy(destination + tail_elements, source,
                head_elements * sizeof(float));
  }
#else
  for (int64_t destination_frequency = 0; destination_frequency < frequencies;
       ++destination_frequency) {
    const int64_t source_frequency =
        (destination_frequency + offset) % frequencies;
    std::copy_n(source + source_frequency * channels, channels,
                destination + destination_frequency * channels);
  }
#endif
  return Ort::Status{nullptr};
}

Ort::Status UnpackFibreBiasAdd(
    const Ort::Custom::Tensor<float>& residual,
    const Ort::Custom::Tensor<float>& projected,
    const Ort::Custom::Tensor<float>& bias,
    const Ort::Custom::Tensor<int64_t>& group_input,
    const Ort::Custom::Tensor<int64_t>& offset_input,
    Ort::Custom::Tensor<float>& output) {
  const auto& shape = residual.Shape();
  if (shape.size() < 2 || group_input.NumberOfElement() != 1 ||
      offset_input.NumberOfElement() != 1) {
    return Ort::Status("invalid Fibre unpack interface", ORT_INVALID_ARGUMENT);
  }
  const int64_t frequencies = shape[shape.size() - 2];
  const int64_t channels = shape.back();
  const int64_t group = group_input.Data()[0];
  const int64_t offset = offset_input.Data()[0];
  if (frequencies <= 0 || channels <= 0 || group <= 0 ||
      frequencies % group != 0 || offset < 0 || offset >= frequencies ||
      residual.NumberOfElement() != frequencies * channels ||
      projected.NumberOfElement() != frequencies * channels ||
      bias.NumberOfElement() != group * channels) {
    return Ort::Status("unsupported Fibre unpack shape", ORT_INVALID_ARGUMENT);
  }
  const float* residual_data = residual.Data();
  const float* projected_data = projected.Data();
  const float* bias_data = bias.Data();
  float* destination = output.Allocate(shape);
  for (int64_t frequency = 0; frequency < frequencies; ++frequency) {
    const int64_t projected_frequency =
        (frequency - offset + frequencies) % frequencies;
    const float* residual_row = residual_data + frequency * channels;
    const float* projected_row = projected_data + projected_frequency * channels;
    const float* bias_row =
        bias_data + (projected_frequency % group) * channels;
    float* destination_row = destination + frequency * channels;
    int64_t channel = 0;
#if defined(__aarch64__) && SETRAIN_ARM_OPTIMIZED_FIBRE_LAYOUT
    for (; channel + 8 <= channels; channel += 8) {
      const float32x4_t residual0 = vld1q_f32(residual_row + channel);
      const float32x4_t projected0 = vld1q_f32(projected_row + channel);
      const float32x4_t bias0 = vld1q_f32(bias_row + channel);
      const float32x4_t residual1 = vld1q_f32(residual_row + channel + 4);
      const float32x4_t projected1 = vld1q_f32(projected_row + channel + 4);
      const float32x4_t bias1 = vld1q_f32(bias_row + channel + 4);
      vst1q_f32(destination_row + channel,
                vaddq_f32(vaddq_f32(residual0, projected0), bias0));
      vst1q_f32(destination_row + channel + 4,
                vaddq_f32(vaddq_f32(residual1, projected1), bias1));
    }
#endif
    for (; channel < channels; ++channel) {
      destination_row[channel] =
          residual_row[channel] + projected_row[channel] + bias_row[channel];
    }
  }
  return Ort::Status{nullptr};
}

Ort::Status UnpackFibreBiasPositionAdd(
    const Ort::Custom::Tensor<float>& residual,
    const Ort::Custom::Tensor<float>& projected,
    const Ort::Custom::Tensor<float>& bias,
    const Ort::Custom::Tensor<float>& position,
    const Ort::Custom::Tensor<int64_t>& group_input,
    const Ort::Custom::Tensor<int64_t>& offset_input,
    Ort::Custom::Tensor<float>& output) {
  const auto& shape = residual.Shape();
  if (shape.size() < 2 || group_input.NumberOfElement() != 1 ||
      offset_input.NumberOfElement() != 1) {
    return Ort::Status("invalid Fibre fused-unpack interface", ORT_INVALID_ARGUMENT);
  }
  const int64_t frequencies = shape[shape.size() - 2];
  const int64_t channels = shape.back();
  const int64_t group = group_input.Data()[0];
  const int64_t offset = offset_input.Data()[0];
  if (frequencies <= 0 || channels <= 0 || group <= 0 ||
      frequencies % group != 0 || offset < 0 || offset >= frequencies ||
      residual.NumberOfElement() != frequencies * channels ||
      projected.NumberOfElement() != frequencies * channels ||
      bias.NumberOfElement() != group * channels ||
      position.NumberOfElement() != frequencies * channels) {
    return Ort::Status("unsupported Fibre fused-unpack shape", ORT_INVALID_ARGUMENT);
  }
  const float* residual_data = residual.Data();
  const float* projected_data = projected.Data();
  const float* bias_data = bias.Data();
  const float* position_data = position.Data();
  float* destination = output.Allocate(shape);
  for (int64_t frequency = 0; frequency < frequencies; ++frequency) {
    const int64_t projected_frequency =
        (frequency - offset + frequencies) % frequencies;
    const float* residual_row = residual_data + frequency * channels;
    const float* projected_row = projected_data + projected_frequency * channels;
    const float* bias_row =
        bias_data + (projected_frequency % group) * channels;
    const float* position_row = position_data + frequency * channels;
    float* destination_row = destination + frequency * channels;
    int64_t channel = 0;
#if defined(__aarch64__) && SETRAIN_ARM_OPTIMIZED_FIBRE_LAYOUT
    for (; channel + 8 <= channels; channel += 8) {
      float32x4_t sum0 = vaddq_f32(
          vld1q_f32(residual_row + channel),
          vld1q_f32(projected_row + channel));
      sum0 = vaddq_f32(sum0, vld1q_f32(bias_row + channel));
      sum0 = vaddq_f32(sum0, vld1q_f32(position_row + channel));
      float32x4_t sum1 = vaddq_f32(
          vld1q_f32(residual_row + channel + 4),
          vld1q_f32(projected_row + channel + 4));
      sum1 = vaddq_f32(sum1, vld1q_f32(bias_row + channel + 4));
      sum1 = vaddq_f32(sum1, vld1q_f32(position_row + channel + 4));
      vst1q_f32(destination_row + channel, sum0);
      vst1q_f32(destination_row + channel + 4, sum1);
    }
#endif
    for (; channel < channels; ++channel) {
      destination_row[channel] = residual_row[channel] + projected_row[channel] +
                                 bias_row[channel] + position_row[channel];
    }
  }
  return Ort::Status{nullptr};
}

void RetainDomain(Ort::CustomOpDomain&& domain) {
  static std::vector<Ort::CustomOpDomain> domains;
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  domains.push_back(std::move(domain));
}

}

extern "C" size_t SetrainFibreGruPackedBankCount() {
  return FibreGruPackedBankCount().load(std::memory_order_relaxed);
}

extern "C" size_t SetrainFibreGruPackedBankCacheHits() {
  return FibreGruPackedBankCacheHits().load(std::memory_order_relaxed);
}

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(
    OrtSessionOptions* options, const OrtApiBase* api_base) {
  Ort::Global<void>::api_ = api_base->GetApi(ORT_API_VERSION);
  try {
    static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> compress{
        Ort::Custom::CreateLiteCustomOp(
        "CompressComplexSpectrum", "CPUExecutionProvider",
        CompressComplexSpectrum, CompressComplexSpectrumShape)};
    static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> combined{
        Ort::Custom::CreateLiteCustomOp(
        "CompressComplexSpectrumAndOrientedGram", "CPUExecutionProvider",
        CompressComplexSpectrumAndOrientedGram,
        CompressComplexSpectrumAndOrientedGramShape)};
    static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> magnitude{
        Ort::Custom::CreateLiteCustomOp(
        "CompressComplexSpectrumAndMagnitude", "CPUExecutionProvider",
        CompressComplexSpectrumAndMagnitude,
        CompressComplexSpectrumAndMagnitudeShape)};
    static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> fused_frontend{
        Ort::Custom::CreateLiteCustomOp(
            "CompressComplexSpectrumAndOrientedGramConv",
            "CPUExecutionProvider",
            CompressComplexSpectrumAndOrientedGramConv,
            CompressComplexSpectrumAndOrientedGramConvShape)};
    static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> decompress{
        Ort::Custom::CreateLiteCustomOp(
        "ApplyMaskAndDecompress", "CPUExecutionProvider",
        ApplyMaskAndDecompress, ApplyMaskAndDecompressShape)};
    static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> pack{
        Ort::Custom::CreateLiteCustomOp(
            "PackFibre", "CPUExecutionProvider", PackFibre)};
    static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> unpack{
        Ort::Custom::CreateLiteCustomOp(
            "UnpackFibreBiasAdd", "CPUExecutionProvider", UnpackFibreBiasAdd)};
    static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> fused_unpack{
        Ort::Custom::CreateLiteCustomOp(
            "UnpackFibreBiasPositionAdd", "CPUExecutionProvider",
            UnpackFibreBiasPositionAdd)};
    static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> fibre_gru{
        Ort::Custom::CreateLiteCustomOp<FibreGru>(
            "FibreGRU", "CPUExecutionProvider")};
#if SETRAIN_ARM_FIBRE_GRU_USE_KLEIDIAI
    static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> fibre_gru_block{
        Ort::Custom::CreateLiteCustomOp<FibreGruBlock>(
            "FibreGRUBlock", "CPUExecutionProvider")};
    static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp>
        fastenhancer_gru_block{
            Ort::Custom::CreateLiteCustomOp<FastEnhancerGruBlock>(
                "FastEnhancerGRUBlock", "CPUExecutionProvider")};
#endif
    Ort::CustomOpDomain domain{kDomain};
    domain.Add(compress.get());
    domain.Add(combined.get());
    domain.Add(magnitude.get());
    domain.Add(fused_frontend.get());
    domain.Add(decompress.get());
    domain.Add(pack.get());
    domain.Add(unpack.get());
    domain.Add(fused_unpack.get());
    domain.Add(fibre_gru.get());
#if SETRAIN_ARM_FIBRE_GRU_USE_KLEIDIAI
    domain.Add(fibre_gru_block.get());
    domain.Add(fastenhancer_gru_block.get());
#endif
    Ort::UnownedSessionOptions(options).Add(domain);
    RetainDomain(std::move(domain));
    return nullptr;
  } catch (const std::exception& error) {
    return Ort::GetApi().CreateStatus(ORT_FAIL, error.what());
  }
}

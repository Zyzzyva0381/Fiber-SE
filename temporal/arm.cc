#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.h"

#if B553_ARM_DOTPROD
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod.h"
#else
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h"
#endif

namespace {

volatile float g_sink = 0.0F;

__attribute__((noinline)) void Consume(float first, float second, float third) {
  g_sink = g_sink + (first + second + third) * 1.0e-30F;
}

struct KaiW8Matmul {
  KaiW8Matmul(const float* source, int input, int output, int rows)
      : input_(input), output_(output), rows_(rows) {
    constexpr size_t kMr = 4;
    constexpr size_t kNr = 4;
#if B553_ARM_DOTPROD
    constexpr size_t kKr = 4;
#else
    constexpr size_t kKr = 8;
#endif
    constexpr size_t kSr = 1;
    std::vector<int8_t> quantized(static_cast<size_t>(output * input));
    std::vector<float> scales(static_cast<size_t>(output));
    std::vector<float> biases(static_cast<size_t>(output), 0.0F);
    for (int out = 0; out < output; ++out) {
      float maximum = 0.0F;
      for (int inner = 0; inner < input; ++inner) {
        maximum = std::max(maximum, std::abs(source[out * input + inner]));
      }
      const float scale = maximum > 0.0F ? maximum / 127.0F : 1.0F;
      scales[out] = scale;
      for (int inner = 0; inner < input; ++inner) {
        const long rounded = std::lrint(source[out * input + inner] / scale);
        quantized[out * input + inner] = static_cast<int8_t>(
            std::max<long>(-127, std::min<long>(127, rounded)));
      }
    }
    rhs_.resize(kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
        output, input, kNr, kKr, kSr));
    const kai_rhs_pack_qsi8cx_params parameters{1, 1.0F};
    kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
        1, output, input, kNr, kKr, kSr, quantized.data(), biases.data(),
        scales.data(), rhs_.data(), 0, &parameters);
    lhs_.resize(kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(
        rows, input, kMr, kKr, kSr));
    output_values_.resize(static_cast<size_t>(rows * output));
  }

  void Quantize(const float* input) {
    constexpr size_t kMr = 4;
#if B553_ARM_DOTPROD
    constexpr size_t kKr = 4;
#else
    constexpr size_t kKr = 8;
#endif
    constexpr size_t kSr = 1;
    kai_run_lhs_quant_pack_qai8dxp_f32(
        rows_, input_, kMr, kKr, kSr, 0, input, input_ * sizeof(float),
        lhs_.data());
  }

  void Dot() {
#if B553_ARM_DOTPROD
    kai_run_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod(
#else
    kai_run_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm(
#endif
        rows_, output_, input_, lhs_.data(), rhs_.data(), output_values_.data(),
        output_ * sizeof(float), sizeof(float),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity());
  }

  void Run(const float* input) {
    Quantize(input);
    Dot();
  }

  float First() const { return output_values_.front(); }
  size_t PackedWeightBytes() const { return rhs_.size(); }

 private:
  int input_{};
  int output_{};
  int rows_{};
  std::vector<uint8_t> rhs_;
  std::vector<uint8_t> lhs_;
  std::vector<float> output_values_;
};

template <class Function>
double Measure(Function&& function, int warmup, int iterations) {
  for (int index = 0; index < warmup; ++index) function();
  const auto begin = std::chrono::steady_clock::now();
  for (int index = 0; index < iterations; ++index) function();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::micro>(end - begin).count()
      / iterations;
}

void RunShape(
    int p, int hidden, int rows, int warmup, int iterations, int repeats) {
  std::mt19937 generator(20260826 + p);
  std::uniform_real_distribution<float> distribution(-0.25F, 0.25F);
  auto values = [&](size_t count) {
    std::vector<float> result(count);
    for (float& value : result) value = distribution(generator);
    return result;
  };
  const auto input0 = values(static_cast<size_t>(rows * hidden));
  const auto input1 = values(static_cast<size_t>(rows * hidden));
  const auto input2 = values(static_cast<size_t>(rows * hidden));
  const auto weight0 = values(static_cast<size_t>(3 * hidden * hidden));
  const auto weight1 = values(static_cast<size_t>(3 * hidden * hidden));
  const auto weight2 = values(static_cast<size_t>(hidden * hidden));
  KaiW8Matmul w80(weight0.data(), hidden, 3 * hidden, rows);
  KaiW8Matmul w81(weight1.data(), hidden, 3 * hidden, rows);
  KaiW8Matmul w82(weight2.data(), hidden, hidden, rows);
  w80.Quantize(input0.data());
  w81.Quantize(input1.data());
  w82.Quantize(input2.data());

  for (int repeat = 0; repeat < repeats; ++repeat) {
    const double quantize = Measure([&] {
      w80.Quantize(input0.data());
      w81.Quantize(input1.data());
      w82.Quantize(input2.data());
      Consume(input0.front(), input1.front(), input2.front());
    }, warmup, iterations);
    const double dot = Measure([&] {
      w80.Dot();
      w81.Dot();
      w82.Dot();
      Consume(w80.First(), w81.First(), w82.First());
    }, warmup, iterations);
    const double full = Measure([&] {
      w80.Run(input0.data());
      w81.Run(input1.data());
      w82.Run(input2.data());
      Consume(w80.First(), w81.First(), w82.First());
    }, warmup, iterations);
    std::cout << "{\"p\":" << p
              << ",\"H\":" << hidden
              << ",\"M\":" << rows
              << ",\"repeat\":" << repeat + 1
              << ",\"w8_quantize_us\":" << quantize
              << ",\"w8_dot_us\":" << dot
              << ",\"w8_full_us\":" << full
              << ",\"packed_weight_bytes\":"
              << w80.PackedWeightBytes() + w81.PackedWeightBytes()
                     + w82.PackedWeightBytes()
              << "}\n";
  }
}

}

int main(int argc, char** argv) {
  int warmup = 2000;
  int iterations = 20000;
  int repeats = 10;
  int selected_p = 0;
  if (argc > 1) warmup = std::stoi(argv[1]);
  if (argc > 2) iterations = std::stoi(argv[2]);
  if (argc > 3) repeats = std::stoi(argv[3]);
  if (argc > 4) selected_p = std::stoi(argv[4]);
  if (selected_p == 0 || selected_p == 1)
    RunShape(1, 80, 16, warmup, iterations, repeats);
  if (selected_p == 0 || selected_p == 2)
    RunShape(2, 112, 8, warmup, iterations, repeats);
  if (selected_p == 0 || selected_p == 4)
    RunShape(4, 160, 4, warmup, iterations, repeats);
  if (selected_p == 0 || selected_p == 8)
    RunShape(8, 224, 2, warmup, iterations, repeats);
  if (selected_p == 0 || selected_p == 16)
    RunShape(16, 320, 1, warmup, iterations, repeats);
  return g_sink == 12345.0F ? 1 : 0;
}

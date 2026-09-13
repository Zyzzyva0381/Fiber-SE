#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

volatile float g_sink = 0.0F;

__attribute__((noinline)) void Consume(float first, float second, float third) {
  g_sink = g_sink + (first + second + third) * 1.0e-30F;
}

struct Fp32Matmul {
  Fp32Matmul(const float* source, int64_t input, int64_t output)
      : input_(input), output_(output), output_padded_((output + 15) & ~15) {
    packed_.assign(static_cast<size_t>(input_ * output_padded_), 0.0F);
    for (int64_t output_begin = 0; output_begin < output_padded_;
         output_begin += 16) {
      for (int64_t inner = 0; inner < input_; ++inner) {
        for (int64_t lane = 0; lane < 16; ++lane) {
          const int64_t out = output_begin + lane;
          if (out < output_) {
            packed_[static_cast<size_t>(
                output_begin * input_ + inner * 16 + lane)] =
                source[out * input_ + inner];
          }
        }
      }
    }
  }

  void Run(const float* values, int64_t rows, float* output) const {
    constexpr int64_t kRowTile = 4;
    constexpr int64_t kOutputTile = 4;
    for (int64_t row_begin = 0; row_begin < rows; row_begin += kRowTile) {
      const int64_t active_rows = std::min<int64_t>(kRowTile, rows - row_begin);
      for (int64_t output_begin = 0; output_begin < output_padded_;
           output_begin += 16 * kOutputTile) {
        const int64_t active_blocks = std::min<int64_t>(
            kOutputTile, (output_padded_ - output_begin) / 16);
        __m512 sums[kOutputTile][kRowTile];
        for (int64_t block = 0; block < active_blocks; ++block) {
          for (int64_t row = 0; row < active_rows; ++row) {
            sums[block][row] = _mm512_setzero_ps();
          }
        }
        for (int64_t inner = 0; inner < input_; ++inner) {
          __m512 weights[kOutputTile];
          for (int64_t block = 0; block < active_blocks; ++block) {
            const float* weight = packed_.data() + static_cast<size_t>(
                (output_begin + block * 16) * input_);
            weights[block] = _mm512_loadu_ps(weight + inner * 16);
          }
          for (int64_t row = 0; row < active_rows; ++row) {
            const __m512 activation = _mm512_set1_ps(
                values[(row_begin + row) * input_ + inner]);
            for (int64_t block = 0; block < active_blocks; ++block) {
              sums[block][row] = _mm512_fmadd_ps(
                  activation, weights[block], sums[block][row]);
            }
          }
        }
        for (int64_t block = 0; block < active_blocks; ++block) {
          const int64_t block_output = output_begin + block * 16;
          const int64_t remaining = output_ - block_output;
          const __mmask16 mask = remaining >= 16
              ? static_cast<__mmask16>(0xFFFF)
              : static_cast<__mmask16>(
                    (1U << std::max<int64_t>(remaining, 0)) - 1U);
          for (int64_t row = 0; row < active_rows; ++row) {
            _mm512_mask_storeu_ps(
                output + (row_begin + row) * output_ + block_output,
                mask, sums[block][row]);
          }
        }
      }
    }
  }

  int64_t input_{};
  int64_t output_{};
  int64_t output_padded_{};
  std::vector<float> packed_;
};

struct W8Matmul {
  W8Matmul(const float* source, int64_t input, int64_t output)
      : input_(input), input_padded_((input + 3) & ~3), output_(output),
        output_padded_((output + 15) & ~15) {
    packed_.assign(
        static_cast<size_t>(output_padded_ * input_padded_), int8_t{0});
    scales_.assign(static_cast<size_t>(output_padded_), 1.0F);
    correction_.assign(static_cast<size_t>(output_padded_), 0);
    for (int64_t out = 0; out < output_; ++out) {
      float maximum = 0.0F;
      for (int64_t inner = 0; inner < input_; ++inner) {
        maximum = std::max(
            maximum, std::abs(source[out * input_ + inner]));
      }
      const float scale = maximum > 0.0F ? maximum / 127.0F : 1.0F;
      scales_[out] = scale;
      int32_t sum = 0;
      for (int64_t inner = 0; inner < input_; ++inner) {
        const long rounded = std::lrint(
            source[out * input_ + inner] / scale);
        const int8_t quantized = static_cast<int8_t>(
            std::max<long>(-127, std::min<long>(127, rounded)));
        const int64_t block = out / 16;
        const int64_t lane = out % 16;
        const int64_t group = inner / 4;
        const int64_t sub = inner % 4;
        packed_[static_cast<size_t>(
            block * input_padded_ * 16 + group * 64 + lane * 4 + sub)] =
            quantized;
        sum += static_cast<int32_t>(quantized);
      }
      correction_[out] = -128 * sum;
    }
  }

  void Quantize(const float* values, int64_t rows) {
    quantized_.resize(static_cast<size_t>(rows * input_padded_));
    activation_scales_.resize(static_cast<size_t>(rows));
    for (int64_t row = 0; row < rows; ++row) {
      const float* source = values + row * input_;
      __m512 vector_maximum = _mm512_setzero_ps();
      int64_t inner = 0;
      for (; inner + 16 <= input_; inner += 16) {
        const __m512 value = _mm512_loadu_ps(source + inner);
        const __m512 absolute = _mm512_andnot_ps(
            _mm512_set1_ps(-0.0F), value);
        vector_maximum = _mm512_max_ps(vector_maximum, absolute);
      }
      float maximum = _mm512_reduce_max_ps(vector_maximum);
      for (; inner < input_; ++inner) {
        maximum = std::max(maximum, std::abs(source[inner]));
      }
      const float scale = maximum > 0.0F ? maximum / 127.0F : 1.0F;
      activation_scales_[row] = scale;
      uint8_t* destination = quantized_.data() + row * input_padded_;
      const __m512 multiplier = _mm512_set1_ps(1.0F / scale);
      const __m512i lower = _mm512_set1_epi32(-127);
      const __m512i upper = _mm512_set1_epi32(127);
      inner = 0;
      for (; inner + 16 <= input_; inner += 16) {
        __m512i integers = _mm512_cvtps_epi32(
            _mm512_mul_ps(_mm512_loadu_ps(source + inner), multiplier));
        integers = _mm512_max_epi32(lower, _mm512_min_epi32(upper, integers));
        __m128i bytes = _mm512_cvtsepi32_epi8(integers);
        bytes = _mm_xor_si128(bytes, _mm_set1_epi8(static_cast<char>(0x80)));
        _mm_storeu_si128(
            reinterpret_cast<__m128i*>(destination + inner), bytes);
      }
      for (; inner < input_; ++inner) {
        const long rounded = std::lrint(source[inner] / scale);
        const long signed_value =
            std::max<long>(-127, std::min<long>(127, rounded));
        destination[inner] = static_cast<uint8_t>(signed_value + 128);
      }
      std::fill(
          destination + input_, destination + input_padded_, uint8_t{128});
    }
  }

  void Dot(int64_t rows, float* output) const {
    constexpr int64_t kRowTile = 4;
    constexpr int64_t kOutputTile = 4;
    for (int64_t row_begin = 0; row_begin < rows; row_begin += kRowTile) {
      const int64_t active_rows = std::min<int64_t>(kRowTile, rows - row_begin);
      for (int64_t output_begin = 0; output_begin < output_padded_;
           output_begin += 16 * kOutputTile) {
        const int64_t active_blocks = std::min<int64_t>(
            kOutputTile, (output_padded_ - output_begin) / 16);
        __m512i sums[kOutputTile][kRowTile];
        for (int64_t block = 0; block < active_blocks; ++block) {
          const __m512i initial = _mm512_loadu_si512(
              correction_.data() + output_begin + block * 16);
          for (int64_t row = 0; row < active_rows; ++row) {
            sums[block][row] = initial;
          }
        }
        for (int64_t inner = 0; inner < input_padded_; inner += 4) {
          __m512i packed_weight[kOutputTile];
          for (int64_t block = 0; block < active_blocks; ++block) {
            const int8_t* weight = packed_.data() + static_cast<size_t>(
                ((output_begin / 16) + block) * input_padded_ * 16);
            packed_weight[block] =
                _mm512_loadu_si512(weight + (inner / 4) * 64);
          }
          for (int64_t row = 0; row < active_rows; ++row) {
            uint32_t word;
            std::memcpy(
                &word,
                quantized_.data() + (row_begin + row) * input_padded_ + inner,
                sizeof(word));
            const __m512i activation =
                _mm512_set1_epi32(static_cast<int32_t>(word));
            for (int64_t block = 0; block < active_blocks; ++block) {
              sums[block][row] = _mm512_dpbusd_epi32(
                  sums[block][row], activation, packed_weight[block]);
            }
          }
        }
        for (int64_t block = 0; block < active_blocks; ++block) {
          const int64_t block_output = output_begin + block * 16;
          const __m512 weight_scales =
              _mm512_loadu_ps(scales_.data() + block_output);
          const int64_t remaining = output_ - block_output;
          const __mmask16 mask = remaining >= 16
              ? static_cast<__mmask16>(0xFFFF)
              : static_cast<__mmask16>(
                    (1U << std::max<int64_t>(remaining, 0)) - 1U);
          for (int64_t row = 0; row < active_rows; ++row) {
            const __m512 result = _mm512_mul_ps(
                _mm512_cvtepi32_ps(sums[block][row]),
                _mm512_mul_ps(
                    weight_scales,
                    _mm512_set1_ps(activation_scales_[row_begin + row])));
            _mm512_mask_storeu_ps(
                output + (row_begin + row) * output_ + block_output,
                mask, result);
          }
        }
      }
    }
  }

  void Run(const float* values, int64_t rows, float* output) {
    Quantize(values, rows);
    Dot(rows, output);
  }

  size_t PackedBytes() const {
    return packed_.size() + scales_.size() * sizeof(float)
        + correction_.size() * sizeof(int32_t);
  }

  float ObservableValue() const {
    return static_cast<float>(quantized_.front()) + activation_scales_.front();
  }

  int64_t input_{};
  int64_t input_padded_{};
  int64_t output_{};
  int64_t output_padded_{};
  std::vector<int8_t> packed_;
  std::vector<float> scales_;
  std::vector<int32_t> correction_;
  std::vector<uint8_t> quantized_;
  std::vector<float> activation_scales_;
};

template <class Function>
double Measure(Function&& function, int iterations) {
  for (int index = 0; index < 20000; ++index) function();
  const auto begin = std::chrono::steady_clock::now();
  for (int index = 0; index < iterations; ++index) function();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::micro>(end - begin).count()
      / iterations;
}

void RunShape(int p, int hidden, int rows, int iterations, int repeats) {
  std::mt19937 generator(20260825 + p);
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
  std::vector<float> output0(static_cast<size_t>(rows * 3 * hidden));
  std::vector<float> output1(static_cast<size_t>(rows * 3 * hidden));
  std::vector<float> output2(static_cast<size_t>(rows * hidden));
  Fp32Matmul fp0(weight0.data(), hidden, 3 * hidden);
  Fp32Matmul fp1(weight1.data(), hidden, 3 * hidden);
  Fp32Matmul fp2(weight2.data(), hidden, hidden);
  W8Matmul w80(weight0.data(), hidden, 3 * hidden);
  W8Matmul w81(weight1.data(), hidden, 3 * hidden);
  W8Matmul w82(weight2.data(), hidden, hidden);
  w80.Quantize(input0.data(), rows);
  w81.Quantize(input1.data(), rows);
  w82.Quantize(input2.data(), rows);

  for (int repeat = 0; repeat < repeats; ++repeat) {
    const double fp32 = Measure([&] {
      fp0.Run(input0.data(), rows, output0.data());
      fp1.Run(input1.data(), rows, output1.data());
      fp2.Run(input2.data(), rows, output2.data());
      Consume(
          output0[repeat % output0.size()], output1[repeat % output1.size()],
          output2[repeat % output2.size()]);
    }, iterations);
    const double quantize = Measure([&] {
      w80.Quantize(input0.data(), rows);
      w81.Quantize(input1.data(), rows);
      w82.Quantize(input2.data(), rows);
      Consume(
          w80.ObservableValue(), w81.ObservableValue(),
          w82.ObservableValue());
    }, iterations);
    const double dot = Measure([&] {
      w80.Dot(rows, output0.data());
      w81.Dot(rows, output1.data());
      w82.Dot(rows, output2.data());
      Consume(
          output0[repeat % output0.size()], output1[repeat % output1.size()],
          output2[repeat % output2.size()]);
    }, iterations);
    const double w8 = Measure([&] {
      w80.Run(input0.data(), rows, output0.data());
      w81.Run(input1.data(), rows, output1.data());
      w82.Run(input2.data(), rows, output2.data());
      Consume(
          output0[repeat % output0.size()], output1[repeat % output1.size()],
          output2[repeat % output2.size()]);
    }, iterations);
    std::cout << "{\"p\":" << p
              << ",\"H\":" << hidden
              << ",\"M\":" << rows
              << ",\"repeat\":" << repeat + 1
              << ",\"fp32_affine_us\":" << fp32
              << ",\"w8_quantize_us\":" << quantize
              << ",\"w8_dot_us\":" << dot
              << ",\"w8_full_us\":" << w8
              << ",\"packed_weight_bytes\":"
              << w80.PackedBytes() + w81.PackedBytes() + w82.PackedBytes()
              << "}\n";
  }
}

}

int main(int argc, char** argv) {
  int iterations = 200000;
  int repeats = 10;
  if (argc > 1) iterations = std::stoi(argv[1]);
  if (argc > 2) repeats = std::stoi(argv[2]);
  RunShape(1, 40, 16, iterations, repeats);
  RunShape(2, 56, 8, iterations, repeats);
  RunShape(4, 80, 4, iterations, repeats);
  RunShape(8, 112, 2, iterations, repeats);
  return g_sink == 12345.0F ? 1 : 0;
}

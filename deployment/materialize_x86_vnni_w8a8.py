from __future__ import annotations

import argparse
from pathlib import Path

from deployment.materialize_x86_fp32 import DEFAULT_SOURCE, materialize as materialize_fp32


W8_MATMUL = r'''struct X86PackedMatmul {
  X86PackedMatmul(
      const float* source, int64_t input, int64_t output,
      bool source_is_output_major, const float* bias = nullptr)
      : input_(input), input_padded_((input + 3) & ~3), output_(output),
        output_padded_((output + 15) & ~15) {
    packed_.assign(
        static_cast<size_t>(output_padded_ * input_padded_), int8_t{0});
    scales_.assign(static_cast<size_t>(output_padded_), 1.0F);
    correction_.assign(static_cast<size_t>(output_padded_), 0);
    bias_.assign(static_cast<size_t>(output_padded_), 0.0F);
    if (bias != nullptr) std::copy(bias, bias + output_, bias_.begin());
    for (int64_t out = 0; out < output_; ++out) {
      float maximum = 0.0F;
      for (int64_t inner = 0; inner < input_; ++inner) {
        const float value = source_is_output_major
            ? source[out * input_ + inner]
            : source[inner * output_ + out];
        maximum = std::max(maximum, std::abs(value));
      }
      const float scale = maximum > 0.0F ? maximum / 127.0F : 1.0F;
      scales_[out] = scale;
      int32_t sum = 0;
      for (int64_t inner = 0; inner < input_; ++inner) {
        const float value = source_is_output_major
            ? source[out * input_ + inner]
            : source[inner * output_ + out];
        const long rounded = std::lrint(value / scale);
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

  void Run(const float* values, int64_t rows, float* output) const {
    constexpr int64_t kRowTile = 4;
    constexpr int64_t kOutputTile = 4;
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
        const long signed_value = std::max<long>(-127, std::min<long>(127, rounded));
        destination[inner] = static_cast<uint8_t>(signed_value + 128);
      }
      std::fill(destination + input_, destination + input_padded_, uint8_t{128});
    }

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
          const __m512 biases = _mm512_loadu_ps(bias_.data() + block_output);
          const int64_t remaining = output_ - block_output;
          const __mmask16 mask = remaining >= 16
              ? static_cast<__mmask16>(0xFFFF)
              : static_cast<__mmask16>(
                    (1U << std::max<int64_t>(remaining, 0)) - 1U);
          for (int64_t row = 0; row < active_rows; ++row) {
            __m512 result = _mm512_mul_ps(
                _mm512_cvtepi32_ps(sums[block][row]),
                _mm512_mul_ps(
                    weight_scales,
                    _mm512_set1_ps(activation_scales_[row_begin + row])));
            result = _mm512_add_ps(result, biases);
            _mm512_mask_storeu_ps(
                output + (row_begin + row) * output_ + block_output,
                mask, result);
          }
        }
      }
    }
  }

  int64_t input_{};
  int64_t input_padded_{};
  int64_t output_{};
  int64_t output_padded_{};
  std::vector<int8_t> packed_;
  std::vector<float> scales_;
  std::vector<int32_t> correction_;
  std::vector<float> bias_;
  mutable std::vector<uint8_t> quantized_;
  mutable std::vector<float> activation_scales_;
};

'''


def materialize(source: Path) -> str:
    text = materialize_fp32(source)
    begin = text.index("struct X86PackedMatmul {")
    end = text.index("inline __m512 X86FastExp", begin)
    return text[:begin] + W8_MATMUL + text[end:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(materialize(args.source), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

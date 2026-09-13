from __future__ import annotations

import argparse
from pathlib import Path

from deployment.materialize_x86_fp32 import DEFAULT_SOURCE, materialize as materialize_fp32


AMX_MATMUL = r'''inline void X86EnableAmxForCurrentThread() {
  thread_local const bool enabled = []() {
    constexpr unsigned long kArchReqXcompPerm = 0x1023;
    constexpr unsigned long kXfeatureTileData = 18;
    if (syscall(
            SYS_arch_prctl, kArchReqXcompPerm, kXfeatureTileData) != 0) {
      throw std::runtime_error("AMX XTILEDATA permission request failed");
    }
    return true;
  }();
  (void)enabled;
}

struct alignas(64) X86AmxTileConfig {
  uint8_t palette_id{};
  uint8_t start_row{};
  uint8_t reserved[14]{};
  uint16_t colsb[16]{};
  uint8_t rows[16]{};
};

struct X86PackedMatmul {
  X86PackedMatmul(
      const float* source, int64_t input, int64_t output,
      bool source_is_output_major, const float* bias = nullptr)
      : input_(input), input_padded_((input + 63) & ~63), output_(output),
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
    X86EnableAmxForCurrentThread();
    quantized_.resize(static_cast<size_t>(rows * input_padded_));
    activation_scales_.resize(static_cast<size_t>(rows));
    for (int64_t row = 0; row < rows; ++row) {
      const float* source = values + row * input_;
      float maximum = 0.0F;
      int64_t inner = 0;
      for (; inner + 16 <= input_; inner += 16) {
        const __m512 value = _mm512_loadu_ps(source + inner);
        const __m512 absolute = _mm512_andnot_ps(
            _mm512_set1_ps(-0.0F), value);
        maximum = std::max(maximum, _mm512_reduce_max_ps(absolute));
      }
      for (; inner < input_; ++inner) maximum = std::max(maximum, std::abs(source[inner]));
      const float scale = maximum > 0.0F ? maximum / 127.0F : 1.0F;
      activation_scales_[row] = scale;
      uint8_t* destination = quantized_.data() + row * input_padded_;
      for (inner = 0; inner < input_; ++inner) {
        const long rounded = std::lrint(source[inner] / scale);
        const long signed_value = std::max<long>(-127, std::min<long>(127, rounded));
        destination[inner] = static_cast<uint8_t>(signed_value + 128);
      }
      std::fill(destination + input_, destination + input_padded_, uint8_t{128});
    }

    alignas(64) int32_t accumulators[16 * 16];
    for (int64_t row_begin = 0; row_begin < rows; row_begin += 16) {
      const int64_t active_rows = std::min<int64_t>(16, rows - row_begin);
      X86AmxTileConfig config{};
      config.palette_id = 1;
      config.colsb[0] = 64;
      config.rows[0] = static_cast<uint8_t>(active_rows);
      config.colsb[1] = 64;
      config.rows[1] = static_cast<uint8_t>(active_rows);
      config.colsb[2] = 64;
      config.rows[2] = 16;
      _tile_loadconfig(&config);
      for (int64_t output_begin = 0; output_begin < output_padded_;
           output_begin += 16) {
        _tile_zero(0);
        const int8_t* weight = packed_.data() +
            static_cast<size_t>((output_begin / 16) * input_padded_ * 16);
        for (int64_t inner = 0; inner < input_padded_; inner += 64) {
          _tile_loadd(
              1,
              quantized_.data() + row_begin * input_padded_ + inner,
              input_padded_);
          _tile_loadd(2, weight + (inner / 4) * 64, 64);
          _tile_dpbusd(0, 1, 2);
        }
        _tile_stored(0, accumulators, 16 * sizeof(int32_t));
        const int64_t remaining = output_ - output_begin;
        const __mmask16 mask = remaining >= 16
            ? static_cast<__mmask16>(0xFFFF)
            : static_cast<__mmask16>((1U << std::max<int64_t>(remaining, 0)) - 1U);
        const __m512i correction =
            _mm512_loadu_si512(correction_.data() + output_begin);
        const __m512 weight_scales = _mm512_loadu_ps(scales_.data() + output_begin);
        const __m512 biases = _mm512_loadu_ps(bias_.data() + output_begin);
        for (int64_t row = 0; row < active_rows; ++row) {
          const __m512i corrected = _mm512_add_epi32(
              _mm512_loadu_si512(accumulators + row * 16), correction);
          __m512 result = _mm512_mul_ps(
              _mm512_cvtepi32_ps(corrected),
              _mm512_mul_ps(
                  weight_scales,
                  _mm512_set1_ps(activation_scales_[row_begin + row])));
          result = _mm512_add_ps(result, biases);
          _mm512_mask_storeu_ps(
              output + (row_begin + row) * output_ + output_begin,
              mask, result);
        }
      }
      _tile_release();
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
    text = text.replace(
        "#include <immintrin.h>\n",
        "#include <immintrin.h>\n#include <sys/syscall.h>\n#include <unistd.h>\n",
        1,
    )
    begin = text.index("struct X86PackedMatmul {")
    end = text.index("inline __m512 X86FastExp", begin)
    return text[:begin] + AMX_MATMUL + text[end:]


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

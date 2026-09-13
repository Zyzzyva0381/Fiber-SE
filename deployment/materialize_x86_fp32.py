from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT / "deployment" / "runtime_base.cc"
)


X86_BLOCKS = r'''
#else

#ifndef SETRAIN_X86_FP32_OUTPUT_TILES
#define SETRAIN_X86_FP32_OUTPUT_TILES 4
#endif

struct X86PackedMatmul {
  X86PackedMatmul(
      const float* source, int64_t input, int64_t output,
      bool source_is_output_major, const float* bias = nullptr)
      : input_(input), output_(output), output_padded_((output + 15) & ~15) {
    packed_.assign(static_cast<size_t>(input_ * output_padded_), 0.0F);
    bias_.assign(static_cast<size_t>(output_padded_), 0.0F);
    if (bias != nullptr) {
      std::copy(bias, bias + output_, bias_.begin());
    }
    for (int64_t output_begin = 0; output_begin < output_padded_;
         output_begin += 16) {
      for (int64_t inner = 0; inner < input_; ++inner) {
        for (int64_t lane = 0; lane < 16; ++lane) {
          const int64_t out = output_begin + lane;
          if (out >= output_) continue;
          packed_[static_cast<size_t>(
              output_begin * input_ + inner * 16 + lane)] =
              source_is_output_major
                  ? source[out * input_ + inner]
                  : source[inner * output_ + out];
        }
      }
    }
  }

  void Run(const float* values, int64_t rows, float* output) const {
    constexpr int64_t kRowTile = 4;
    constexpr int64_t kOutputTile = SETRAIN_X86_FP32_OUTPUT_TILES;
    for (int64_t row_begin = 0; row_begin < rows; row_begin += kRowTile) {
      const int64_t active_rows = std::min<int64_t>(kRowTile, rows - row_begin);
      for (int64_t output_begin = 0; output_begin < output_padded_;
           output_begin += 16 * kOutputTile) {
        const int64_t active_blocks = std::min<int64_t>(
            kOutputTile, (output_padded_ - output_begin) / 16);
        __m512 sums[kOutputTile][kRowTile];
        for (int64_t block = 0; block < active_blocks; ++block) {
          const __m512 initial = _mm512_loadu_ps(
              bias_.data() + output_begin + block * 16);
          for (int64_t row = 0; row < active_rows; ++row) {
            sums[block][row] = initial;
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
  std::vector<float> bias_;
};

inline __m512 X86FastExp(__m512 value) {
  const __m512 log2e = _mm512_set1_ps(1.4426950408889634F);
  const __m512 ln2 = _mm512_set1_ps(0.6931471805599453F);
  value = _mm512_max_ps(
      _mm512_set1_ps(-18.0F),
      _mm512_min_ps(_mm512_set1_ps(18.0F), value));
  const __m512 rounded = _mm512_roundscale_ps(
      _mm512_mul_ps(value, log2e),
      _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  const __m512 remainder =
      _mm512_fnmadd_ps(rounded, ln2, value);
  __m512 polynomial = _mm512_set1_ps(1.0F / 720.0F);
  polynomial = _mm512_fmadd_ps(
      polynomial, remainder, _mm512_set1_ps(1.0F / 120.0F));
  polynomial = _mm512_fmadd_ps(
      polynomial, remainder, _mm512_set1_ps(1.0F / 24.0F));
  polynomial = _mm512_fmadd_ps(
      polynomial, remainder, _mm512_set1_ps(1.0F / 6.0F));
  polynomial = _mm512_fmadd_ps(
      polynomial, remainder, _mm512_set1_ps(0.5F));
  polynomial = _mm512_fmadd_ps(
      polynomial, remainder, _mm512_set1_ps(1.0F));
  polynomial = _mm512_fmadd_ps(
      polynomial, remainder, _mm512_set1_ps(1.0F));
  const __m512i exponent = _mm512_cvtps_epi32(rounded);
  const __m512i exponent_bits = _mm512_slli_epi32(
      _mm512_add_epi32(exponent, _mm512_set1_epi32(127)), 23);
  return _mm512_mul_ps(polynomial, _mm512_castsi512_ps(exponent_bits));
}

inline __m512 X86Logistic(__m512 value) {
  return _mm512_div_ps(
      _mm512_set1_ps(1.0F),
      _mm512_add_ps(_mm512_set1_ps(1.0F), X86FastExp(_mm512_sub_ps(
          _mm512_setzero_ps(), value))));
}

inline __m512 X86Tanh(__m512 value) {
  return _mm512_sub_ps(
      _mm512_mul_ps(
          _mm512_set1_ps(2.0F),
          X86Logistic(_mm512_mul_ps(_mm512_set1_ps(2.0F), value))),
      _mm512_set1_ps(1.0F));
}

struct X86GruBank {
  X86GruBank(const float* input, const float* recurrent, int64_t hidden)
      : hidden(hidden),
        input_affine(input, hidden, 3 * hidden, true),
        recurrent_affine(recurrent, hidden, 3 * hidden, true) {}
  int64_t hidden{};
  X86PackedMatmul input_affine;
  X86PackedMatmul recurrent_affine;
};

std::shared_ptr<const X86GruBank> PackX86GruBank(
    const float* input_weight, const float* recurrent_weight, int64_t hidden) {
  const size_t elements = static_cast<size_t>(3 * hidden * hidden);
  uint64_t key = HashFloats(input_weight, elements);
  key ^= HashFloats(recurrent_weight, elements) +
         0x9e3779b97f4a7c15ULL + (key << 6) + (key >> 2);
  key ^= static_cast<uint64_t>(hidden) * 0xbf58476d1ce4e5b9ULL;
  static std::mutex mutex;
  static std::unordered_map<uint64_t, std::weak_ptr<const X86GruBank>> cache;
  std::lock_guard<std::mutex> lock(mutex);
  if (const auto found = cache.find(key); found != cache.end()) {
    if (auto bank = found->second.lock()) {
      FibreGruPackedBankCacheHits().fetch_add(1, std::memory_order_relaxed);
      return bank;
    }
  }
  auto bank = std::make_shared<X86GruBank>(input_weight, recurrent_weight, hidden);
  cache[key] = bank;
  FibreGruPackedBankCount().fetch_add(1, std::memory_order_relaxed);
  return bank;
}

void RunX86Gru(
    const X86GruBank& bank, const std::vector<float>& bias,
    const float* input, const float* initial, int64_t batch,
    std::vector<float>& input_affine, std::vector<float>& recurrent_affine,
    float* output) {
  const int64_t hidden = bank.hidden;
  input_affine.resize(static_cast<size_t>(batch * 3 * hidden));
  recurrent_affine.resize(static_cast<size_t>(batch * 3 * hidden));
  bank.input_affine.Run(input, batch, input_affine.data());
  bank.recurrent_affine.Run(initial, batch, recurrent_affine.data());
  for (int64_t row = 0; row < batch; ++row) {
    const float* x = input_affine.data() + row * 3 * hidden;
    const float* h = recurrent_affine.data() + row * 3 * hidden;
    int64_t index = 0;
    for (; index + 16 <= hidden; index += 16) {
      const __m512 z = X86Logistic(_mm512_add_ps(
          _mm512_add_ps(_mm512_loadu_ps(x + index),
                        _mm512_loadu_ps(h + index)),
          _mm512_add_ps(_mm512_loadu_ps(bias.data() + index),
                        _mm512_loadu_ps(bias.data() + 3 * hidden + index))));
      const __m512 r = X86Logistic(_mm512_add_ps(
          _mm512_add_ps(_mm512_loadu_ps(x + hidden + index),
                        _mm512_loadu_ps(h + hidden + index)),
          _mm512_add_ps(_mm512_loadu_ps(bias.data() + hidden + index),
                        _mm512_loadu_ps(bias.data() + 4 * hidden + index))));
      const __m512 candidate = X86Tanh(_mm512_add_ps(
          _mm512_add_ps(_mm512_loadu_ps(x + 2 * hidden + index),
                        _mm512_loadu_ps(bias.data() + 2 * hidden + index)),
          _mm512_mul_ps(r, _mm512_add_ps(
              _mm512_loadu_ps(h + 2 * hidden + index),
              _mm512_loadu_ps(bias.data() + 5 * hidden + index)))));
      const __m512 old_state = _mm512_loadu_ps(initial + row * hidden + index);
      _mm512_storeu_ps(
          output + row * hidden + index,
          _mm512_fmadd_ps(
              z, _mm512_sub_ps(old_state, candidate), candidate));
    }
    for (; index < hidden; ++index) {
      const float z = Logistic(
          x[index] + h[index] + bias[index] + bias[3 * hidden + index]);
      const float r = Logistic(
          x[hidden + index] + h[hidden + index] +
          bias[hidden + index] + bias[4 * hidden + index]);
      const float candidate = std::tanh(
          x[2 * hidden + index] + bias[2 * hidden + index] +
          r * (h[2 * hidden + index] + bias[5 * hidden + index]));
      output[row * hidden + index] =
          candidate + z * (initial[row * hidden + index] - candidate);
    }
  }
}

struct FibreGruBlock {
  FibreGruBlock(const OrtApi*, const OrtKernelInfo* raw_info) {
    const Ort::ConstKernelInfo info{raw_info};
    auto constant = [&info](size_t index) {
      int is_constant = 0;
      Ort::ConstValue value = info.GetTensorConstantInput(index, &is_constant);
      if (!is_constant) throw std::runtime_error("FibreGRUBlock requires constants");
      return value;
    };
    const auto w = constant(2); const auto r = constant(3); const auto b = constant(4);
    const auto pw = constant(6); const auto pb = constant(7); const auto pos = constant(8);
    const auto group = constant(9); const auto offset = constant(10);
    const auto shape = w.GetTensorTypeAndShapeInfo().GetShape();
    hidden_ = shape.size() == 3 ? shape[2] : 0;
    batch_ = hidden_ ? static_cast<int64_t>(pos.GetTensorTypeAndShapeInfo().GetElementCount()) / hidden_ : 0;
    if (hidden_ <= 0 || batch_ <= 0 || shape != std::vector<int64_t>({1, 3 * hidden_, hidden_}) ||
        r.GetTensorTypeAndShapeInfo().GetShape() != shape ||
        b.GetTensorTypeAndShapeInfo().GetElementCount() != static_cast<size_t>(6 * hidden_) ||
        pw.GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>({hidden_, hidden_}) ||
        pb.GetTensorTypeAndShapeInfo().GetElementCount() != static_cast<size_t>(hidden_)) {
      throw std::runtime_error("unsupported FibreGRUBlock shape");
    }
    bank_ = PackX86GruBank(w.GetTensorData<float>(), r.GetTensorData<float>(), hidden_);
    bias_.assign(b.GetTensorData<float>(), b.GetTensorData<float>() + 6 * hidden_);
    projection_ = std::make_unique<X86PackedMatmul>(
        pw.GetTensorData<float>(), hidden_, hidden_, false, pb.GetTensorData<float>());
    group_ = group.GetTensorData<int64_t>()[0]; offset_ = offset.GetTensorData<int64_t>()[0];
  }

  Ort::Status Compute(
      const Ort::Custom::Tensor<float>& input, const Ort::Custom::Tensor<float>& residual,
      const Ort::Custom::Tensor<float>&, const Ort::Custom::Tensor<float>&,
      const Ort::Custom::Tensor<float>&, const Ort::Custom::Tensor<float>& initial,
      const Ort::Custom::Tensor<float>&, const Ort::Custom::Tensor<float>&,
      const Ort::Custom::Tensor<float>& position, const Ort::Custom::Tensor<int64_t>&,
      const Ort::Custom::Tensor<int64_t>&, Ort::Custom::Tensor<float>& output,
      Ort::Custom::Tensor<float>& final) {
    const auto& shape = residual.Shape();
    if (shape.size() < 2) return Ort::Status("invalid Fibre residual", ORT_INVALID_ARGUMENT);
    const int64_t frequencies = shape[shape.size() - 2], channels = shape.back();
    if (group_ <= 0 || frequencies / group_ != batch_ || channels * group_ != hidden_ ||
        input.NumberOfElement() != batch_ * hidden_ || initial.NumberOfElement() != batch_ * hidden_) {
      return Ort::Status("invalid FibreGRUBlock input", ORT_INVALID_ARGUMENT);
    }
    const float* gru_input = input.Data();
    if (offset_ != 0) {
      reordered_.resize(static_cast<size_t>(batch_ * hidden_));
      const int64_t first_frequencies = group_ - offset_;
      const int64_t first_columns = first_frequencies * channels;
      const int64_t second_columns = offset_ * channels;
      for (int64_t row = 0; row < batch_; ++row) {
        const int64_t first_frequency = offset_ + row * group_;
        const int64_t second_frequency = (first_frequency + first_frequencies) % frequencies;
        float* destination = reordered_.data() + row * hidden_;
        std::memcpy(destination, input.Data() + first_frequency * channels,
                    static_cast<size_t>(first_columns) * sizeof(float));
        std::memcpy(destination + first_columns, input.Data() + second_frequency * channels,
                    static_cast<size_t>(second_columns) * sizeof(float));
      }
      gru_input = reordered_.data();
    }
    float* state = final.Allocate({1, batch_, hidden_});
    RunX86Gru(*bank_, bias_, gru_input, initial.Data(), batch_, x_affine_, h_affine_, state);
    projected_.resize(static_cast<size_t>(batch_ * hidden_));
    projection_->Run(state, batch_, projected_.data());
    float* destination = output.Allocate(shape);
    for (int64_t frequency = 0; frequency < frequencies; ++frequency) {
      const int64_t packed_frequency = (frequency - offset_ + frequencies) % frequencies;
      for (int64_t channel = 0; channel < channels; ++channel) {
        const int64_t out_index = frequency * channels + channel;
        destination[out_index] = residual.Data()[out_index] + position.Data()[out_index] +
            projected_[packed_frequency * channels + channel];
      }
    }
    return Ort::Status{nullptr};
  }
  static Ort::Status InferOutputShape(Ort::ShapeInferContext& context) {
    context.SetOutputShape(0, context.GetInputShape(1));
    context.SetOutputShape(1, context.GetInputShape(5));
    return Ort::Status{nullptr};
  }
  int64_t hidden_{}, batch_{}, group_{}, offset_{};
  std::shared_ptr<const X86GruBank> bank_;
  std::vector<float> bias_, reordered_, x_affine_, h_affine_, projected_;
  std::unique_ptr<X86PackedMatmul> projection_;
};

struct FastEnhancerGruBlock {
  FastEnhancerGruBlock(const OrtApi*, const OrtKernelInfo* raw_info) {
    const Ort::ConstKernelInfo info{raw_info};
    auto constant = [&info](size_t index) {
      int is_constant = 0; Ort::ConstValue value = info.GetTensorConstantInput(index, &is_constant);
      if (!is_constant) throw std::runtime_error("FastEnhancerGRUBlock requires constants");
      return value;
    };
    const auto w = constant(2); const auto r = constant(3); const auto b = constant(4);
    const auto pw = constant(6); const auto pb = constant(7); const auto pos = constant(8);
    const auto shape = w.GetTensorTypeAndShapeInfo().GetShape();
    hidden_ = shape.size() == 3 ? shape[2] : 0;
    batch_ = hidden_ ? static_cast<int64_t>(pos.GetTensorTypeAndShapeInfo().GetElementCount()) / hidden_ : 0;
    if (hidden_ <= 0 || batch_ <= 0 || shape != std::vector<int64_t>({1, 3 * hidden_, hidden_}) ||
        r.GetTensorTypeAndShapeInfo().GetShape() != shape ||
        b.GetTensorTypeAndShapeInfo().GetElementCount() != static_cast<size_t>(6 * hidden_) ||
        pw.GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>({hidden_, hidden_}) ||
        pb.GetTensorTypeAndShapeInfo().GetElementCount() != static_cast<size_t>(hidden_)) {
      throw std::runtime_error("unsupported FastEnhancerGRUBlock shape");
    }
    bank_ = PackX86GruBank(w.GetTensorData<float>(), r.GetTensorData<float>(), hidden_);
    bias_.assign(b.GetTensorData<float>(), b.GetTensorData<float>() + 6 * hidden_);
    projection_ = std::make_unique<X86PackedMatmul>(
        pw.GetTensorData<float>(), hidden_, hidden_, false, pb.GetTensorData<float>());
  }
  Ort::Status Compute(
      const Ort::Custom::Tensor<float>& input, const Ort::Custom::Tensor<float>& residual,
      const Ort::Custom::Tensor<float>&, const Ort::Custom::Tensor<float>&,
      const Ort::Custom::Tensor<float>&, const Ort::Custom::Tensor<float>& initial,
      const Ort::Custom::Tensor<float>&, const Ort::Custom::Tensor<float>&,
      const Ort::Custom::Tensor<float>& position, Ort::Custom::Tensor<float>& output,
      Ort::Custom::Tensor<float>& final) {
    if (input.NumberOfElement() != batch_ * hidden_ || residual.NumberOfElement() != batch_ * hidden_ ||
        initial.NumberOfElement() != batch_ * hidden_ || position.NumberOfElement() != batch_ * hidden_) {
      return Ort::Status("invalid FastEnhancerGRUBlock input", ORT_INVALID_ARGUMENT);
    }
    float* state = final.Allocate({1, batch_, hidden_});
    RunX86Gru(*bank_, bias_, input.Data(), initial.Data(), batch_, x_affine_, h_affine_, state);
    projected_.resize(static_cast<size_t>(batch_ * hidden_));
    projection_->Run(state, batch_, projected_.data());
    float* destination = output.Allocate(residual.Shape());
    for (int64_t index = 0; index < batch_ * hidden_; ++index) {
      destination[index] = projected_[index] + residual.Data()[index] + position.Data()[index];
    }
    return Ort::Status{nullptr};
  }
  static Ort::Status InferOutputShape(Ort::ShapeInferContext& context) {
    context.SetOutputShape(0, context.GetInputShape(1));
    context.SetOutputShape(1, context.GetInputShape(5));
    return Ort::Status{nullptr};
  }
  int64_t hidden_{}, batch_{};
  std::shared_ptr<const X86GruBank> bank_;
  std::vector<float> bias_, x_affine_, h_affine_, projected_;
  std::unique_ptr<X86PackedMatmul> projection_;
};

struct X86FibreSpectralAttentionBlock {
  X86FibreSpectralAttentionBlock(const OrtApi*, const OrtKernelInfo* raw_info) {
    const Ort::ConstKernelInfo info{raw_info};
    auto constant = [&info](size_t index) {
      int is_constant = 0;
      Ort::ConstValue value = info.GetTensorConstantInput(index, &is_constant);
      if (!is_constant) {
        throw std::runtime_error(
            "FibreSpectralAttentionBlock requires constant projections");
      }
      return value;
    };
    const auto qkv = constant(1);
    const auto projection = constant(2);
    const auto bias = constant(3);
    const auto qkv_shape = qkv.GetTensorTypeAndShapeInfo().GetShape();
    if (qkv_shape.size() != 2 || qkv_shape[1] != 3 * qkv_shape[0] ||
        qkv_shape[0] <= 0 || qkv_shape[0] % heads_ != 0) {
      throw std::runtime_error("unsupported Fibre spectral QKV shape");
    }
    channels_ = qkv_shape[0];
    head_channels_ = channels_ / heads_;
    if (projection.GetTensorTypeAndShapeInfo().GetShape() !=
            std::vector<int64_t>({channels_, channels_}) ||
        bias.GetTensorTypeAndShapeInfo().GetElementCount() !=
            static_cast<size_t>(channels_)) {
      throw std::runtime_error("unsupported Fibre spectral projection shape");
    }
    qkv_ = std::make_unique<X86PackedMatmul>(
        qkv.GetTensorData<float>(), channels_, 3 * channels_, false);
    projection_ = std::make_unique<X86PackedMatmul>(
        projection.GetTensorData<float>(), channels_, channels_, false,
        bias.GetTensorData<float>());
  }

  Ort::Status Compute(
      const Ort::Custom::Tensor<float>& input,
      const Ort::Custom::Tensor<float>&,
      const Ort::Custom::Tensor<float>&,
      const Ort::Custom::Tensor<float>&,
      Ort::Custom::Tensor<float>& output) {
    const auto& shape = input.Shape();
    if (shape.size() < 2 || shape.back() != channels_ ||
        input.NumberOfElement() % channels_ != 0) {
      return Ort::Status(
          "invalid FibreSpectralAttentionBlock input", ORT_INVALID_ARGUMENT);
    }
    const int64_t frequencies = input.NumberOfElement() / channels_;
    if (frequencies <= 0 || frequencies > 64) {
      return Ort::Status(
          "unsupported Fibre attention frequency count", ORT_INVALID_ARGUMENT);
    }

    qkv_values_.resize(
        static_cast<size_t>(frequencies * 3 * channels_));
    attended_.assign(
        static_cast<size_t>(frequencies * channels_), 0.0F);
    projected_.resize(static_cast<size_t>(frequencies * channels_));
    scores_.resize(static_cast<size_t>(frequencies));
    probabilities_.resize(static_cast<size_t>(frequencies));
    qkv_->Run(input.Data(), frequencies, qkv_values_.data());

    const float scale = 1.0F / std::sqrt(static_cast<float>(head_channels_));
    for (int64_t head = 0; head < heads_; ++head) {
      const int64_t q_offset = head * 3 * head_channels_;
      const int64_t k_offset = q_offset + head_channels_;
      const int64_t v_offset = k_offset + head_channels_;
      for (int64_t query = 0; query < frequencies; ++query) {
        const float* q = qkv_values_.data() +
            query * 3 * channels_ + q_offset;
        float maximum = -std::numeric_limits<float>::infinity();
        for (int64_t key = 0; key < frequencies; ++key) {
          const float* k = qkv_values_.data() +
              key * 3 * channels_ + k_offset;
          float score = 0.0F;
          for (int64_t channel = 0; channel < head_channels_; ++channel) {
            score += q[channel] * k[channel];
          }
          score *= scale;
          scores_[static_cast<size_t>(key)] = score;
          maximum = std::max(maximum, score);
        }
        float denominator = 0.0F;
        for (int64_t key = 0; key < frequencies; ++key) {
          const float probability =
              std::exp(scores_[static_cast<size_t>(key)] - maximum);
          probabilities_[static_cast<size_t>(key)] = probability;
          denominator += probability;
        }
        const float reciprocal = 1.0F / denominator;
        float* destination = attended_.data() +
            query * channels_ + head * head_channels_;
        for (int64_t key = 0; key < frequencies; ++key) {
          const float probability =
              probabilities_[static_cast<size_t>(key)] * reciprocal;
          const float* value = qkv_values_.data() +
              key * 3 * channels_ + v_offset;
          for (int64_t channel = 0; channel < head_channels_; ++channel) {
            destination[channel] += probability * value[channel];
          }
        }
      }
    }

    projection_->Run(attended_.data(), frequencies, projected_.data());
    float* destination = output.Allocate(shape);
    const int64_t elements = frequencies * channels_;
    int64_t index = 0;
    for (; index + 16 <= elements; index += 16) {
      _mm512_storeu_ps(
          destination + index,
          _mm512_add_ps(
              _mm512_loadu_ps(input.Data() + index),
              _mm512_loadu_ps(projected_.data() + index)));
    }
    for (; index < elements; ++index) {
      destination[index] = input.Data()[index] + projected_[index];
    }
    return Ort::Status{nullptr};
  }

  static Ort::Status InferOutputShape(Ort::ShapeInferContext& context) {
    context.SetOutputShape(0, context.GetInputShape(0));
    return Ort::Status{nullptr};
  }

  static constexpr int64_t heads_ = 4;
  int64_t channels_{}, head_channels_{};
  std::unique_ptr<X86PackedMatmul> qkv_, projection_;
  std::vector<float> qkv_values_, attended_, projected_, scores_, probabilities_;
};

struct X86FibreConv1dQuickGelu {
  X86FibreConv1dQuickGelu(const OrtApi*, const OrtKernelInfo* raw_info) {
    const Ort::ConstKernelInfo info{raw_info};
    stride_ = info.GetAttribute<int64_t>("stride");
    pad_ = info.GetAttribute<int64_t>("pad");
    int is_constant = 0;
    const auto weight = info.GetTensorConstantInput(1, &is_constant);
    if (!is_constant) {
      throw std::runtime_error("FibreConv1dQuickGelu requires constant weights");
    }
    const auto weight_shape = weight.GetTensorTypeAndShapeInfo().GetShape();
    if (weight_shape.size() != 3 || weight_shape[0] <= 0 ||
        weight_shape[1] <= 0 || weight_shape[2] <= 0 || stride_ <= 0 ||
        pad_ < 0) {
      throw std::runtime_error("unsupported FibreConv1dQuickGelu shape");
    }
    output_channels_ = weight_shape[0];
    input_channels_ = weight_shape[1];
    kernel_ = weight_shape[2];
    const auto bias = info.GetTensorConstantInput(2, &is_constant);
    if (!is_constant || bias.GetTensorTypeAndShapeInfo().GetElementCount() !=
            static_cast<size_t>(output_channels_)) {
      throw std::runtime_error("FibreConv1dQuickGelu requires constant bias");
    }
    convolution_ = std::make_unique<X86PackedMatmul>(
        weight.GetTensorData<float>(), input_channels_ * kernel_,
        output_channels_, true, bias.GetTensorData<float>());
    constexpr int64_t kMaximumPositions = 256;
    patches_.resize(static_cast<size_t>(
        kMaximumPositions * input_channels_ * kernel_));
    affine_.resize(static_cast<size_t>(kMaximumPositions * output_channels_));
  }

  Ort::Status Compute(
      const Ort::Custom::Tensor<float>& input,
      const Ort::Custom::Tensor<float>&,
      const Ort::Custom::Tensor<float>&,
      Ort::Custom::Tensor<float>& output) {
    const auto& shape = input.Shape();
    if (shape.size() != 3 || shape[0] != 1 || shape[1] != input_channels_) {
      return Ort::Status(
          "invalid FibreConv1dQuickGelu input", ORT_INVALID_ARGUMENT);
    }
    const int64_t input_positions = shape[2];
    const int64_t output_positions =
        (input_positions + 2 * pad_ - kernel_) / stride_ + 1;
    if (output_positions <= 0 || output_positions > 256) {
      return Ort::Status(
          "unsupported FibreConv1dQuickGelu positions", ORT_INVALID_ARGUMENT);
    }
    const int64_t columns = input_channels_ * kernel_;
    for (int64_t position = 0; position < output_positions; ++position) {
      float* patch = patches_.data() + position * columns;
      for (int64_t channel = 0; channel < input_channels_; ++channel) {
        for (int64_t tap = 0; tap < kernel_; ++tap) {
          const int64_t source_position = position * stride_ + tap - pad_;
          patch[channel * kernel_ + tap] =
              source_position >= 0 && source_position < input_positions
                  ? input.Data()[channel * input_positions + source_position]
                  : 0.0F;
        }
      }
    }
    convolution_->Run(patches_.data(), output_positions, affine_.data());
    float* destination = output.Allocate(
        {1, output_channels_, output_positions});
    for (int64_t position = 0; position < output_positions; ++position) {
      const float* source = affine_.data() + position * output_channels_;
      int64_t channel = 0;
      for (; channel + 16 <= output_channels_; channel += 16) {
        const __m512 value = _mm512_loadu_ps(source + channel);
        const __m512 activated = _mm512_mul_ps(value, X86Logistic(value));
        alignas(64) float lanes[16];
        _mm512_store_ps(lanes, activated);
        for (int64_t lane = 0; lane < 16; ++lane) {
          destination[(channel + lane) * output_positions + position] = lanes[lane];
        }
      }
      for (; channel < output_channels_; ++channel) {
        const float value = source[channel];
        destination[channel * output_positions + position] =
            value / (1.0F + std::exp(-value));
      }
    }
    return Ort::Status{nullptr};
  }

  static Ort::Status InferOutputShape(Ort::ShapeInferContext& context) {
    const auto input = context.GetInputShape(0);
    const auto weight = context.GetInputShape(1);
    if (input.size() != 3 || weight.size() != 3) {
      return Ort::Status("invalid Fibre Conv shape", ORT_INVALID_ARGUMENT);
    }
    if (!input[0].IsInt() || !input[2].IsInt() || !weight[0].IsInt() ||
        !weight[2].IsInt()) {
      return Ort::Status("dynamic Fibre Conv shape", ORT_INVALID_ARGUMENT);
    }
    const int64_t stride = context.GetAttrInt("stride");
    const int64_t pad = context.GetAttrInt("pad");
    const int64_t positions =
        (input[2].AsInt() + 2 * pad - weight[2].AsInt()) / stride + 1;
    context.SetOutputShape(
        0, {input[0].AsInt(), weight[0].AsInt(), positions});
    return Ort::Status{nullptr};
  }

  int64_t input_channels_{}, output_channels_{}, kernel_{}, stride_{}, pad_{};
  std::unique_ptr<X86PackedMatmul> convolution_;
  std::vector<float> patches_, affine_;
};

struct X86FibreConv1dK3WinogradQuickGelu {
  X86FibreConv1dK3WinogradQuickGelu(
      const OrtApi*, const OrtKernelInfo* raw_info) {
    const Ort::ConstKernelInfo info{raw_info};
    int is_constant = 0;
    const auto weight = info.GetTensorConstantInput(1, &is_constant);
    if (!is_constant) {
      throw std::runtime_error("Fibre Winograd requires constant weights");
    }
    const auto shape = weight.GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 3 || shape[0] <= 0 || shape[1] <= 0 || shape[2] != 3) {
      throw std::runtime_error("unsupported Fibre Winograd weight shape");
    }
    output_channels_ = shape[0];
    input_channels_ = shape[1];
    const auto bias = info.GetTensorConstantInput(2, &is_constant);
    if (!is_constant || bias.GetTensorTypeAndShapeInfo().GetElementCount() !=
            static_cast<size_t>(output_channels_)) {
      throw std::runtime_error("Fibre Winograd requires constant bias");
    }
    bias_.assign(
        bias.GetTensorData<float>(),
        bias.GetTensorData<float>() + output_channels_);
    const float* source = weight.GetTensorData<float>();
    std::vector<float> transformed[4];
    for (auto& values : transformed) {
      values.resize(static_cast<size_t>(output_channels_ * input_channels_));
    }
    for (int64_t out = 0; out < output_channels_; ++out) {
      for (int64_t in = 0; in < input_channels_; ++in) {
        const float* taps = source + (out * input_channels_ + in) * 3;
        const size_t index = static_cast<size_t>(out * input_channels_ + in);
        transformed[0][index] = taps[0];
        transformed[1][index] = 0.5F * (taps[0] + taps[1] + taps[2]);
        transformed[2][index] = 0.5F * (taps[0] - taps[1] + taps[2]);
        transformed[3][index] = taps[2];
      }
    }
    for (int component = 0; component < 4; ++component) {
      convolution_[component] = std::make_unique<X86PackedMatmul>(
          transformed[component].data(), input_channels_, output_channels_, true);
    }
    constexpr int64_t kMaximumTiles = 128;
    for (int component = 0; component < 4; ++component) {
      transformed_input_[component].resize(
          static_cast<size_t>(kMaximumTiles * input_channels_));
      transformed_output_[component].resize(
          static_cast<size_t>(kMaximumTiles * output_channels_));
    }
  }

  Ort::Status Compute(
      const Ort::Custom::Tensor<float>& input,
      const Ort::Custom::Tensor<float>&,
      const Ort::Custom::Tensor<float>&,
      Ort::Custom::Tensor<float>& output) {
    const auto& shape = input.Shape();
    if (shape.size() != 3 || shape[0] != 1 || shape[1] != input_channels_ ||
        shape[2] <= 0 || shape[2] % 2 != 0 || shape[2] > 256) {
      return Ort::Status("invalid Fibre Winograd input", ORT_INVALID_ARGUMENT);
    }
    const int64_t positions = shape[2];
    const int64_t tiles = positions / 2;
    for (int64_t tile = 0; tile < tiles; ++tile) {
      const int64_t p0 = 2 * tile - 1;
      const int64_t p1 = 2 * tile;
      const int64_t p2 = 2 * tile + 1;
      const int64_t p3 = 2 * tile + 2;
      for (int64_t channel = 0; channel < input_channels_; ++channel) {
        const float* row = input.Data() + channel * positions;
        const float x0 = p0 >= 0 ? row[p0] : 0.0F;
        const float x1 = row[p1];
        const float x2 = row[p2];
        const float x3 = p3 < positions ? row[p3] : 0.0F;
        const size_t index = static_cast<size_t>(tile * input_channels_ + channel);
        transformed_input_[0][index] = x0 - x2;
        transformed_input_[1][index] = x1 + x2;
        transformed_input_[2][index] = x2 - x1;
        transformed_input_[3][index] = x1 - x3;
      }
    }
    for (int component = 0; component < 4; ++component) {
      convolution_[component]->Run(
          transformed_input_[component].data(), tiles,
          transformed_output_[component].data());
    }
    float* destination = output.Allocate({1, output_channels_, positions});
    for (int64_t tile = 0; tile < tiles; ++tile) {
      for (int64_t channel = 0; channel < output_channels_; ++channel) {
        const size_t index = static_cast<size_t>(tile * output_channels_ + channel);
        const float y0 = transformed_output_[0][index] +
            transformed_output_[1][index] + transformed_output_[2][index] +
            bias_[static_cast<size_t>(channel)];
        const float y1 = transformed_output_[1][index] -
            transformed_output_[2][index] - transformed_output_[3][index] +
            bias_[static_cast<size_t>(channel)];
        destination[channel * positions + 2 * tile] =
            y0 / (1.0F + std::exp(-y0));
        destination[channel * positions + 2 * tile + 1] =
            y1 / (1.0F + std::exp(-y1));
      }
    }
    return Ort::Status{nullptr};
  }

  static Ort::Status InferOutputShape(Ort::ShapeInferContext& context) {
    context.SetOutputShape(0, context.GetInputShape(0));
    return Ort::Status{nullptr};
  }

  int64_t input_channels_{}, output_channels_{};
  std::vector<float> bias_;
  std::unique_ptr<X86PackedMatmul> convolution_[4];
  std::vector<float> transformed_input_[4], transformed_output_[4];
};

'''


def materialize(source: Path) -> str:
    text = source.read_text(encoding="utf-8")
    include_marker = "#include <vector>\n"
    if text.count(include_marker) != 1:
        raise RuntimeError("unexpected include marker")
    text = text.replace(
        include_marker,
        include_marker + "\n#if defined(__x86_64__)\n#include <immintrin.h>\n#endif\n",
        1,
    )
    block_end = "};\n\n#endif\n\nOrt::Status CompressComplexSpectrum("
    if text.count(block_end) != 1:
        raise RuntimeError("unexpected symmetric block boundary")
    text = text.replace(
        block_end,
        "};\n\n" + X86_BLOCKS + "#endif\n\nOrt::Status CompressComplexSpectrum(",
        1,
    )
    text = text.replace(
        "#if SETRAIN_ARM_FIBRE_GRU_USE_KLEIDIAI\n    static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> fibre_gru_block",
        "#if SETRAIN_ARM_FIBRE_GRU_USE_KLEIDIAI || defined(__x86_64__)\n    static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> fibre_gru_block",
        1,
    )
    text = text.replace(
        "#if SETRAIN_ARM_FIBRE_GRU_USE_KLEIDIAI\n    domain.Add(fibre_gru_block.get());",
        "#if SETRAIN_ARM_FIBRE_GRU_USE_KLEIDIAI || defined(__x86_64__)\n    domain.Add(fibre_gru_block.get());",
        1,
    )
    definition_marker = (
        "#endif\n    Ort::CustomOpDomain domain{kDomain};"
    )
    if text.count(definition_marker) != 1:
        raise RuntimeError("unexpected custom-op definition boundary")
    text = text.replace(
        definition_marker,
        "#if defined(__x86_64__)\n"
        "    static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> "
        "fibre_spectral_attention_block{\n"
        "        Ort::Custom::CreateLiteCustomOp<X86FibreSpectralAttentionBlock>(\n"
        "            \"FibreSpectralAttentionBlock\", \"CPUExecutionProvider\")};\n"
        "    static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> "
        "fibre_conv1d_quick_gelu{\n"
        "        Ort::Custom::CreateLiteCustomOp<X86FibreConv1dQuickGelu>(\n"
        "            \"FibreConv1dQuickGelu\", \"CPUExecutionProvider\")};\n"
        "    static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> "
        "fibre_conv1d_k3_winograd_quick_gelu{\n"
        "        Ort::Custom::CreateLiteCustomOp<"
        "X86FibreConv1dK3WinogradQuickGelu>(\n"
        "            \"FibreConv1dK3WinogradQuickGelu\", "
        "\"CPUExecutionProvider\")};\n"
        "#endif\n"
        + definition_marker,
        1,
    )
    registration_marker = (
        "    domain.Add(fastenhancer_gru_block.get());\n#endif\n"
    )
    if text.count(registration_marker) != 1:
        raise RuntimeError("unexpected custom-op registration boundary")
    text = text.replace(
        registration_marker,
        registration_marker
        + "#if defined(__x86_64__)\n"
        + "    domain.Add(fibre_spectral_attention_block.get());\n"
        + "    domain.Add(fibre_conv1d_quick_gelu.get());\n"
        + "    domain.Add(fibre_conv1d_k3_winograd_quick_gelu.get());\n"
        + "#endif\n",
        1,
    )
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(materialize(args.source), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

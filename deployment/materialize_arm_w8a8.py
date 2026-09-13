from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT / "deployment" / "runtime_base.cc"
)


W8_CLASSES = r'''template <int Input, int Output, int Rows>
struct KaiW8Matmul {
  KaiW8Matmul(
      const float* source, bool source_is_output_major,
      const float* bias = nullptr) {
    constexpr size_t kMr = 4;
    constexpr size_t kNr = 4;
    constexpr size_t kKr = 8;
    constexpr size_t kSr = 1;
    std::vector<int8_t> quantized(Output * Input);
    std::vector<float> scales(Output);
    std::vector<float> biases(Output, 0.0F);
    if (bias != nullptr) std::copy(bias, bias + Output, biases.begin());
    for (int output = 0; output < Output; ++output) {
      float maximum = 0.0F;
      for (int input = 0; input < Input; ++input) {
        const float value = source_is_output_major
            ? source[output * Input + input]
            : source[input * Output + output];
        maximum = std::max(maximum, std::abs(value));
      }
      const float scale = maximum > 0.0F ? maximum / 127.0F : 1.0F;
      scales[output] = scale;
      for (int input = 0; input < Input; ++input) {
        const float value = source_is_output_major
            ? source[output * Input + input]
            : source[input * Output + output];
        const long rounded = std::lrint(value / scale);
        quantized[output * Input + input] = static_cast<int8_t>(
            std::max<long>(-127, std::min<long>(127, rounded)));
      }
    }
    rhs_.resize(kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
        Output, Input, kNr, kKr, kSr));
    const kai_rhs_pack_qsi8cx_params params{1, 1.0F};
    kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
        1, Output, Input, kNr, kKr, kSr, quantized.data(), biases.data(),
        scales.data(), rhs_.data(), 0, &params);
    lhs_.resize(kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(
        Rows, Input, kMr, kKr, kSr));
    output_.resize(Rows * Output);
  }

  const float* Run(const float* input) {
    constexpr size_t kMr = 4;
    constexpr size_t kKr = 8;
    constexpr size_t kSr = 1;
    kai_run_lhs_quant_pack_qai8dxp_f32(
        Rows, Input, kMr, kKr, kSr, 0, input, Input * sizeof(float),
        lhs_.data());
    kai_run_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm(
        Rows, Output, Input, lhs_.data(), rhs_.data(), output_.data(),
        Output * sizeof(float), sizeof(float),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity());
    return output_.data();
  }

 private:
  std::vector<uint8_t> rhs_;
  std::vector<uint8_t> lhs_;
  std::vector<float> output_;
};

template <int Hidden, int Batch>
struct FibreGruKaiAffine {
  static constexpr int kHidden = Hidden;
  static constexpr int kBatch = Batch;
  static constexpr int kGates = 3 * Hidden;

  FibreGruKaiAffine(const float* input_weight, const float* recurrent_weight)
      : input_(input_weight, true), recurrent_(recurrent_weight, true) {
    split_input_.resize(kBatch * kHidden);
  }

  std::pair<const float*, const float*> Run(
      const float* input, const float* recurrent) {
    return {input_.Run(input), recurrent_.Run(recurrent)};
  }

  std::pair<const float*, const float*> RunSplitInput(
      const float* const* first_rows, size_t first_columns,
      const float* const* second_rows, size_t second_columns,
      const float* recurrent) {
    if (first_columns + second_columns != kHidden) {
      throw std::runtime_error("invalid W8 split-input width");
    }
    for (int row = 0; row < kBatch; ++row) {
      float* destination = split_input_.data() + row * kHidden;
      std::memcpy(destination, first_rows[row], first_columns * sizeof(float));
      std::memcpy(
          destination + first_columns, second_rows[row],
          second_columns * sizeof(float));
    }
    return Run(split_input_.data(), recurrent);
  }

 private:
  KaiW8Matmul<kHidden, kGates, kBatch> input_;
  KaiW8Matmul<kHidden, kGates, kBatch> recurrent_;
  std::vector<float> split_input_;
};

template <int Hidden, int Batch>
struct FibreProjectionKai {
  FibreProjectionKai(const float* weight, const float* bias)
      : projection_(weight, false, bias) {}
  const float* Run(const float* input) { return projection_.Run(input); }
 private:
  KaiW8Matmul<Hidden, Hidden, Batch> projection_;
};'''


BASE_FIBRE_SHAPES = ((80, 4), (144, 4), (160, 4), (160, 8), (176, 6))


def _shape_member(hidden: int, batch: int) -> str:
    return f"h{hidden}_b{batch}"


def add_fibre_block_shapes(
    text: str, extra_shapes: Iterable[tuple[int, int]]
) -> str:

    extras = tuple(
        shape for shape in dict.fromkeys(extra_shapes) if shape not in BASE_FIBRE_SHAPES
    )
    if not extras:
        return text
    for hidden, batch in extras:
        if hidden <= 0 or batch <= 0 or hidden % 4:
            raise ValueError(f"invalid fixed Fibre shape: {(hidden, batch)}")




    for hidden, base_batch in ((144, 4), (176, 6)):
        if not any(value == hidden and batch != base_batch for value, batch in extras):
            continue
        pack = f"  }} else if (hidden == {hidden}) {{\n"
        if text.count(pack) != 1:
            raise RuntimeError(f"unexpected H{hidden} packed-bank dispatch")
        text = text.replace(
            pack,
            f"  }} else if (hidden == {hidden} && batch_hint == {base_batch}) {{\n",
            1,
        )
        projection = f"    }} else if (hidden_ == {hidden}) {{\n"
        if text.count(projection) == 1:
            text = text.replace(
                projection,
                f"    }} else if (hidden_ == {hidden} && batch_ == {base_batch}) {{\n",
                1,
            )
        compute = f"    if (hidden_ == {hidden}) {{\n"
        if text.count(compute) != 1:
            raise RuntimeError(f"unexpected H{hidden} compute dispatch")
        text = text.replace(
            compute,
            f"    if (hidden_ == {hidden} && batch_ == {base_batch}) {{\n",
            1,
        )

    bank_anchor = "  std::shared_ptr<FibreGruKaiAffine<176, 6>> kai_h176;\n"
    bank_rows = "".join(
        f"  std::shared_ptr<FibreGruKaiAffine<{hidden}, {batch}>> "
        f"kai_{_shape_member(hidden, batch)};\n"
        for hidden, batch in extras
    )
    if text.count(bank_anchor) != 1:
        raise RuntimeError("unexpected FibreGruBank member anchor")
    text = text.replace(bank_anchor, bank_anchor + bank_rows, 1)

    h176_condition = (
        "hidden == 176 && batch_hint == 6"
        if any(hidden == 176 and batch != 6 for hidden, batch in extras)
        else "hidden == 176"
    )
    pack_anchor = (
        f"  }} else if ({h176_condition}) {{\n"
        "    bank->kai_h176 = std::make_shared<FibreGruKaiAffine<176, 6>>(\n"
        "        input_weight, recurrent_weight);\n"
        "  } else {\n"
        "#endif\n"
    )
    pack_rows = "".join(
        f"  }} else if (hidden == {hidden} && batch_hint == {batch}) {{\n"
        f"    bank->kai_{_shape_member(hidden, batch)} = "
        f"std::make_shared<FibreGruKaiAffine<{hidden}, {batch}>>(\n"
        "        input_weight, recurrent_weight);\n"
        for hidden, batch in extras
    )
    if text.count(pack_anchor) != 1:
        raise RuntimeError("unexpected PackFibreGruBank dispatch anchor")
    text = text.replace(
        pack_anchor,
        pack_anchor.replace("  } else {\n#endif\n", pack_rows + "  } else {\n#endif\n"),
        1,
    )

    allowed_anchor = (
        "    if ((hidden_ != 80 && hidden_ != 144 && hidden_ != 160 &&\n"
        "         hidden_ != 176) ||\n"
    )
    allowed = BASE_FIBRE_SHAPES + extras
    allowed_expression = " ||\n            ".join(
        f"(hidden_ == {hidden} && batch_ == {batch})" for hidden, batch in allowed
    )


    batch_anchor = (
        "    hidden_ = weight_shape.size() == 3 ? weight_shape[2] : 0;\n"
        + allowed_anchor
    )
    batch_replacement = (
        "    hidden_ = weight_shape.size() == 3 ? weight_shape[2] : 0;\n"
        "    batch_ = hidden_ > 0 ? static_cast<int64_t>(\n"
        "        position.GetTensorTypeAndShapeInfo().GetElementCount()) / hidden_ : 0;\n"
        f"    if (!({allowed_expression}) ||\n"
    )
    if text.count(batch_anchor) != 1:
        raise RuntimeError("unexpected FibreGRUBlock allow-list anchor")
    text = text.replace(batch_anchor, batch_replacement, 1)
    old_batch = (
        "    batch_ = static_cast<int64_t>(\n"
        "        position.GetTensorTypeAndShapeInfo().GetElementCount()) / hidden_;\n"
    )
    if text.count(old_batch) != 1:
        raise RuntimeError("unexpected FibreGRUBlock batch derivation")
    text = text.replace(old_batch, "", 1)

    projection_anchor = (
        "    } else {\n"
        "      projection_h176_ = std::make_unique<FibreProjectionKai<176, 6>>(\n"
        "          projection_weight.GetTensorData<float>(),\n"
        "          projection_bias.GetTensorData<float>());\n"
        "    }\n"
        "    group_ = group.GetTensorData<int64_t>()[0];\n"
    )
    projection_rows = (
        "    } else if (hidden_ == 176 && batch_ == 6) {\n"
        "      projection_h176_ = std::make_unique<FibreProjectionKai<176, 6>>(\n"
        "          projection_weight.GetTensorData<float>(),\n"
        "          projection_bias.GetTensorData<float>());\n"
        + "".join(
            f"    }} else if (hidden_ == {hidden} && batch_ == {batch}) {{\n"
            f"      projection_{_shape_member(hidden, batch)}_ = "
            f"std::make_unique<FibreProjectionKai<{hidden}, {batch}>>(\n"
            "          projection_weight.GetTensorData<float>(),\n"
            "          projection_bias.GetTensorData<float>());\n"
            for hidden, batch in extras
        )
        + "    } else {\n"
        '      throw std::runtime_error("unsupported FibreGRUBlock projection shape");\n'
        "    }\n"
        "    group_ = group.GetTensorData<int64_t>()[0];\n"
    )
    if text.count(projection_anchor) != 1:
        raise RuntimeError("unexpected FibreGRUBlock projection anchor")
    text = text.replace(projection_anchor, projection_rows, 1)

    compute_anchor = (
        "    return Ort::Status(\"unsupported FibreGRUBlock dispatch\", ORT_INVALID_ARGUMENT);\n"
    )
    compute_rows = "".join(
        f"    if (hidden_ == {hidden} && batch_ == {batch}) {{\n"
        f"      return ComputeFixed<{hidden}, {batch}>(\n"
        "          packed_input, residual, initial, position, output, final,\n"
        f"          gru_bank_->kai_{_shape_member(hidden, batch)}, "
        f"projection_{_shape_member(hidden, batch)}_.get());\n"
        "    }\n"
        for hidden, batch in extras
    )
    if text.count(compute_anchor) != 1:
        raise RuntimeError("unexpected FibreGRUBlock compute anchor")
    text = text.replace(compute_anchor, compute_rows + compute_anchor, 1)

    member_anchor = (
        "  std::unique_ptr<FibreProjectionKai<176, 6>> projection_h176_;\n"
    )
    member_rows = "".join(
        f"  std::unique_ptr<FibreProjectionKai<{hidden}, {batch}>> "
        f"projection_{_shape_member(hidden, batch)}_;\n"
        for hidden, batch in extras
    )
    if text.count(member_anchor) != 1:
        raise RuntimeError("unexpected FibreGRUBlock projection member anchor")
    return text.replace(member_anchor, member_anchor + member_rows, 1)


def materialize(
    source: Path, extra_shapes: Iterable[tuple[int, int]] = ()
) -> str:
    text = source.read_text(encoding="utf-8")
    include_marker = (
        '#include "kai/ukernels/matmul/pack/'
        'kai_rhs_pack_nxk_x32p16x1bx32_x32_x32_neon.h"\n'
    )
    additions = (
        '#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/'
        'kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h"\n'
        '#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"\n'
        '#include "kai/ukernels/matmul/pack/'
        'kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.h"\n'
    )
    if text.count(include_marker) != 1:
        raise RuntimeError("unexpected KleidiAI include marker")
    text = text.replace(include_marker, include_marker + additions, 1)
    begin = text.index("template <int Hidden, int Batch>\nstruct FibreGruKaiAffine {")
    end_marker = "\n#endif\n\nuint64_t HashFloats"
    end = text.index(end_marker, begin)
    text = text[:begin] + W8_CLASSES + text[end:]
    return add_fibre_block_shapes(text, extra_shapes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--extra-shape",
        action="append",
        default=[],
        metavar="HIDDENxBATCH",
        help="add one exploratory fixed FibreGRUBlock KleidiAI dispatch",
    )
    args = parser.parse_args()
    extra_shapes = []
    for value in args.extra_shape:
        try:
            hidden, batch = (int(item) for item in value.lower().split("x", 1))
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid --extra-shape {value!r}; expected HIDDENxBATCH") from error
        extra_shapes.append((hidden, batch))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(materialize(args.source, extra_shapes), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

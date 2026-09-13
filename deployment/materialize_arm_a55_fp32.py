from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from deployment.materialize_arm_w8a8 import add_fibre_block_shapes


GENERIC_KERNEL = "kai_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla"
A55_KERNEL = f"{GENERIC_KERNEL}_cortexa55"

GENERIC_INDIRECT = """    const unsigned int lengths[2] = {
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
"""

A55_CONTIGUOUS = """    std::vector<float> contiguous(kBatch * kHidden);
    for (int row = 0; row < kBatch; ++row) {
      float* destination = contiguous.data() + row * kHidden;
      std::memcpy(destination, first_rows[row], first_columns * sizeof(float));
      std::memcpy(destination + first_columns, second_rows[row],
                  second_columns * sizeof(float));
    }
    Multiply(contiguous.data(), packed_weight, output);
"""


def materialize(
    source: Path, extra_shapes: Iterable[tuple[int, int]] = ()
) -> str:
    text = source.read_text(encoding="utf-8")
    if text.count(GENERIC_INDIRECT) != 1:
        raise RuntimeError("unexpected generic indirect-input implementation")
    text = text.replace(GENERIC_INDIRECT, A55_CONTIGUOUS)
    if GENERIC_KERNEL not in text:
        raise RuntimeError("generic ARM FP32 kernel is absent")
    generic_stem = GENERIC_KERNEL.removeprefix("kai_")
    a55_stem = A55_KERNEL.removeprefix("kai_")
    return add_fibre_block_shapes(
        text.replace(generic_stem, a55_stem), extra_shapes
    )

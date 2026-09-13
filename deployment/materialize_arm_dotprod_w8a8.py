from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from deployment.materialize_arm_w8a8 import materialize as materialize_i8mm


I8MM_KERNEL = "kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm"
DOTPROD_KERNEL = "kai_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod"


def materialize(
    source: Path, extra_shapes: Iterable[tuple[int, int]] = ()
) -> str:
    text = materialize_i8mm(source, extra_shapes)
    i8mm_run = I8MM_KERNEL.replace("kai_", "kai_run_", 1)
    dotprod_run = DOTPROD_KERNEL.replace("kai_", "kai_run_", 1)
    if text.count(I8MM_KERNEL) != 1 or text.count(i8mm_run) != 1:
        raise RuntimeError("unexpected i8mm kernel references")
    text = text.replace(i8mm_run, dotprod_run)
    text = text.replace(I8MM_KERNEL, DOTPROD_KERNEL)
    marker = "constexpr size_t kKr = 8;"
    if text.count(marker) != 2:
        raise RuntimeError("unexpected i8mm packing geometry")
    return text.replace(marker, "constexpr size_t kKr = 4;")

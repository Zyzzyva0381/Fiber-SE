from __future__ import annotations

import argparse
import json
import platform
import shlex
import subprocess
import tempfile
from pathlib import Path

from deployment.materialize_arm_a55_fp32 import materialize as materialize_arm_a55_fp32
from deployment.materialize_arm_dotprod_w8a8 import materialize as materialize_arm_dotprod
from deployment.materialize_arm_w8a8 import (
    add_fibre_block_shapes,
    materialize as materialize_arm_w8a8,
)
from deployment.materialize_x86_amx_w8a8 import materialize as materialize_x86_amx
from deployment.materialize_x86_fp32 import materialize as materialize_x86_fp32
from deployment.materialize_x86_vnni_w8a8 import materialize as materialize_x86_vnni
from deployment.support import ORT_HEADER_NAMES, ensure_ort_headers


ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE = ROOT / "deployment" / "runtime_base.cc"
KLEIDIAI_COMMIT = "6787251d9cc2f38a3a6024b11fd7ace10cde4cd9"
KLEIDIAI_URL = "https://github.com/ARM-software/kleidiai.git"

def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _kleidiai(path: Path | None) -> Path:
    if path is not None:
        return path.resolve()
    cache = Path.home() / ".cache" / "fiber-se" / "kleidiai"
    if not (cache / ".git").is_dir():
        cache.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", KLEIDIAI_URL, str(cache)])
    _run(["git", "-C", str(cache), "checkout", "--detach", KLEIDIAI_COMMIT])
    return cache


def _materialize(
    backend: str,
    output: Path,
    fibre_shapes: tuple[tuple[int, int], ...] = (),
) -> None:
    shape_backends = {
        "arm-fp32",
        "arm-kleidiai-a55-fp32",
        "arm-kleidiai-w8a8",
        "arm-kleidiai-dotprod-w8a8",
    }
    if fibre_shapes and backend not in shape_backends:
        raise ValueError(
            f"--fibre-shape is only valid for {', '.join(sorted(shape_backends))}"
        )
    if backend in {"arm-fp32"}:
        output.write_text(
            add_fibre_block_shapes(
                BASE_SOURCE.read_text(encoding="utf-8"), fibre_shapes
            ),
            encoding="utf-8",
        )
        return
    if backend == "arm-kleidiai-a55-fp32":
        output.write_text(
            materialize_arm_a55_fp32(BASE_SOURCE, fibre_shapes), encoding="utf-8"
        )
        return
    if backend == "arm-kleidiai-w8a8":
        output.write_text(
            materialize_arm_w8a8(BASE_SOURCE, fibre_shapes), encoding="utf-8"
        )
        return
    if backend == "arm-kleidiai-dotprod-w8a8":
        output.write_text(
            materialize_arm_dotprod(BASE_SOURCE, fibre_shapes), encoding="utf-8"
        )
        return
    functions = {
        "x86-fp32": materialize_x86_fp32,
        "x86-vnni-w8a8": materialize_x86_vnni,
        "x86-amx-w8a8": materialize_x86_amx,
    }
    output.write_text(functions[backend](BASE_SOURCE), encoding="utf-8")


def _compile_x86(
    source: Path,
    output: Path,
    headers: Path,
    cxx: str,
    cpu_flags: list[str],
) -> None:
    _run(
        [
            cxx,
            "-std=c++17",
            "-O3",
            *cpu_flags,
            "-fPIC",
            "-shared",
            "-Wall",
            "-Wextra",
            f"-I{headers}",
            str(source),
            "-o",
            str(output),
        ]
    )


def _compile_arm(
    backend: str,
    source: Path,
    output: Path,
    headers: Path,
    kleidiai: Path,
    cc: str,
    cxx: str,
    cpu_flags: list[str],
    build: Path,
) -> None:
    if backend == "arm-fp32":
        sources = (
            "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/"
            "kai_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla.c",
            "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/"
            "kai_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla_asm.S",
            "kai/ukernels/matmul/pack/"
            "kai_rhs_pack_nxk_x32p16x1bx32_x32_x32_neon.c",
            "kai/ukernels/matmul/pack/"
            "kai_rhs_pack_nxk_x32p16x1bx32_x32_x32_neon_asm.S",
        )
    elif backend == "arm-kleidiai-a55-fp32":
        sources = (
            "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/"
            "kai_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla_cortexa55.c",
            "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/"
            "kai_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla_cortexa55_asm.S",
            "kai/ukernels/matmul/pack/"
            "kai_rhs_pack_nxk_x32p16x1bx32_x32_x32_neon.c",
            "kai/ukernels/matmul/pack/"
            "kai_rhs_pack_nxk_x32p16x1bx32_x32_x32_neon_asm.S",
        )
    elif backend == "arm-kleidiai-w8a8":
        sources = (
            "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.c",
            "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.c",
            "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/"
            "kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.c",
            "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/"
            "kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm_asm.S",
        )
    else:
        sources = (
            "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.c",
            "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.c",
            "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/"
            "kai_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod.c",
        )
    objects = []
    for index, relative in enumerate(sources):
        obj = build / f"kai-{index}.o"
        _run(
            [
                cc,
                "-O3",
                *cpu_flags,
                "-fPIC",
                f"-I{kleidiai}",
                "-c",
                str(kleidiai / relative),
                "-o",
                str(obj),
            ]
        )
        objects.append(obj)
    _run(
        [
            cxx,
            "-std=c++17",
            "-O3",
            *cpu_flags,
            "-fPIC",
            "-shared",
            "-Wall",
            "-Wextra",
            "-DSETRAIN_ARM_FIBRE_GRU_USE_KLEIDIAI=1",
            f"-I{headers}",
            f"-I{kleidiai}",
            str(source),
            *(str(path) for path in objects),
            "-o",
            str(output),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        required=True,
        choices=(
            "x86-fp32",
            "x86-vnni-w8a8",
            "x86-amx-w8a8",
            "arm-fp32",
            "arm-kleidiai-w8a8",
            "arm-kleidiai-dotprod-w8a8",
            "arm-kleidiai-a55-fp32",
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kleidiai", type=Path)
    parser.add_argument("--cpu-flags")
    parser.add_argument("--cc", default="gcc")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument(
        "--fibre-shape",
        action="append",
        default=[],
        metavar="HIDDENxBATCH",
        help="add one fixed FibreGRUBlock dispatch to an ARM W8A8 runtime",
    )
    args = parser.parse_args()

    fibre_shapes = []
    for value in args.fibre_shape:
        try:
            hidden, batch = (int(item) for item in value.lower().split("x", 1))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid --fibre-shape {value!r}; expected HIDDENxBATCH"
            ) from error
        if hidden <= 0 or batch <= 0:
            raise ValueError(f"invalid --fibre-shape {value!r}")
        fibre_shapes.append((hidden, batch))
    fibre_shapes = tuple(dict.fromkeys(fibre_shapes))

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    headers = ensure_ort_headers()
    default_flags = "-march=native" if args.backend.startswith("x86-") else "-mcpu=native"
    cpu_flags = shlex.split(args.cpu_flags or default_flags)
    with tempfile.TemporaryDirectory(prefix="fiber-runtime-build.") as temporary:
        build = Path(temporary)
        source = build / "runtime.cc"
        _materialize(args.backend, source, fibre_shapes)
        if args.backend.startswith("x86-"):
            _compile_x86(source, output, headers, args.cxx, cpu_flags)
            kleidiai = None
        else:
            kleidiai = _kleidiai(args.kleidiai)
            _compile_arm(
                args.backend,
                source,
                output,
                headers,
                kleidiai,
                args.cc,
                args.cxx,
                cpu_flags,
                build,
            )
        materialized_source_bytes = source.stat().st_size

    report = {
        "schema": "fiber-se.native-runtime-build.v2",
        "backend": args.backend,
        "machine": platform.machine(),
        "cpu": platform.processor(),
        "cpu_flags": cpu_flags,
        "fibre_shapes": [list(shape) for shape in fibre_shapes],
        "compiler": subprocess.check_output([args.cxx, "--version"], text=True).splitlines()[0],
        "base_source": {"path": str(BASE_SOURCE), "bytes": BASE_SOURCE.stat().st_size},
        "materialized_source": {"bytes": materialized_source_bytes},
        "onnxruntime_headers": [
            {"path": str(headers / name), "bytes": (headers / name).stat().st_size}
            for name in ORT_HEADER_NAMES
        ],
        "kleidiai_commit": KLEIDIAI_COMMIT if kleidiai is not None else None,
        "output": {"path": str(output), "bytes": output.stat().st_size},
    }
    report_path = output.with_suffix(output.suffix + ".build.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

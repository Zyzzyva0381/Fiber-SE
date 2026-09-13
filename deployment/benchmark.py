from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from deployment.support import (
    ORT_VERSION,
    compile_runner,
    ensure_ort_header,
    ort_library,
    write_initializer_sidecar,
)


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "runner.cc"

FP32_PROTOCOL = {
    "Xeon": {"backend": "FUSED-FP32", "graph_variant": "fused-temporal"},
    "AmpereOne": {"backend": "FUSED-FP32", "graph_variant": "fused-temporal"},
    "AMD-Ryzen": {"backend": "FP32", "graph_variant": "native-temporal"},  # does not use FUSED-FP32, for it's slower on this cpu
    "RDK-X5": {"backend": "FUSED-FP32", "graph_variant": "fused-temporal"},
}

SELECTIVE_W8A8_PROTOCOL = {
    "Xeon": ("VNNI-W8A8", "AMX-W8A8"),
    "AmpereOne": ("KleidiAI-i8mm-W8A8",),
    "AMD-Ryzen": ("VNNI-W8A8",),
    "RDK-X5": ("KleidiAI-dotprod-W8A8",),
}

PLATFORM_CANDIDATES = {
    platform: (FP32_PROTOCOL[platform]["backend"], *SELECTIVE_W8A8_PROTOCOL[platform])
    for platform in FP32_PROTOCOL
}

GRAPH_VARIANTS = {
    platform: {
        FP32_PROTOCOL[platform]["backend"]: FP32_PROTOCOL[platform]["graph_variant"],
        **{backend: "fused-temporal" for backend in SELECTIVE_W8A8_PROTOCOL[platform]},
    }
    for platform in FP32_PROTOCOL
}


def validate_candidate(platform: str, backend: str, custom_ops: list[Path]) -> None:
    candidates = PLATFORM_CANDIDATES[platform]
    if backend not in candidates:
        raise ValueError(
            f"{backend!r} is not in the {platform} candidate set: {', '.join(candidates)}"
        )
    if not custom_ops:
        raise ValueError("this optimized candidate requires --custom-op")


def graph_variant(model: Path) -> str:
    import onnx

    graph = onnx.load(str(model), load_external_data=False)
    temporal_ops = {"FibreGRUBlock", "FastEnhancerGRUBlock"}
    if any(node.op_type in temporal_ops for node in graph.graph.node):
        return "fused-temporal"
    if any(node.op_type == "GRU" for node in graph.graph.node):
        return "native-temporal"
    raise ValueError("deployment graph has neither native GRU nor fused temporal blocks")


def validate_graph_variant(platform: str, backend: str, model: Path) -> str:
    actual = graph_variant(model)
    expected = GRAPH_VARIANTS[platform][backend]
    if actual != expected:
        raise ValueError(
            f"{platform}/{backend} requires {expected}, but {model} is {actual}"
        )
    return actual


def _ssh_prefix(host: str, options: list[str]) -> list[str]:
    command = ["ssh"]
    for option in options:
        command.extend(("-o", option))
    command.append(host)
    return command


def _ssh(
    host: str,
    options: list[str],
    command: list[str],
    *,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*_ssh_prefix(host, options), shlex.join(command)],
        check=check,
        text=True,
        capture_output=capture,
    )


def _scp(source: str, destination: str, options: list[str]) -> None:
    command = ["scp"]
    for option in options:
        command.extend(("-o", option))
    subprocess.run([*command, source, destination], check=True)


def command(
    binary: str,
    model: str,
    sidecar: str,
    report: str,
    args: argparse.Namespace,
    custom_ops: list[str] | None = None,
) -> list[str]:
    result = [
        binary,
        "--model", model,
        "--output", report,
        "--cpu", args.cpu,
        "--warmup", str(args.warmup),
        "--iterations", str(args.iterations),
        "--repeats", str(args.repeats),
        "--seed", str(args.seed),
        "--frame-ms", str(args.frame_ms),
    ]
    if getattr(args, "embedded_initializers", False):
        result.append("--embedded-initializers")
    else:
        result.extend(("--initializers", sidecar))
    for library in custom_ops or []:
        result.extend(("--custom-op", library))
    if args.ort_info:
        result.append("--ort-info")
    return result


def finalize(
    report: dict[str, Any],
    *,
    model: Path,
    sidecar_summary: dict[str, Any],
    output: Path,
    remote: str | None,
    custom_ops: list[Path],
    platform: str,
    backend: str,
    header: Path,
    ort_runtime: dict[str, Any],
    measured_graph_variant: str,
) -> dict[str, Any]:
    runtime_sidecar = report["initializer_sidecar"]
    report["initializer_sidecar_runtime"] = {
        "lifetime": "temporary benchmark staging",
        "runtime_path": runtime_sidecar,
    }
    del report["initializer_sidecar"]
    report["source_model"] = str(model.resolve())
    report["initializer_artifact"] = {
        key: value for key, value in sidecar_summary.items() if key != "path"
    }
    report["cpp_source"] = str(SOURCE.resolve())
    report["onnxruntime_c_api_header"] = str(header.resolve())
    report["onnxruntime_library"] = ort_runtime
    report["remote"] = remote
    report["platform"] = platform
    report["backend"] = backend
    report["graph_variant"] = measured_graph_variant
    report["custom_op_libraries"] = [str(path.resolve()) for path in custom_ops]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def benchmark_local(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="fiber-deployment.") as temporary:
        root = Path(temporary)
        sidecar = root / "initializers.bin"
        summary = write_initializer_sidecar(args.model, sidecar)
        header = ensure_ort_header()
        binary = root / "benchmark_deployment"
        library = getattr(args, "ort_library", None) or ort_library(args.runtime_python)
        compile_runner(
            source=SOURCE,
            header=header,
            output=binary,
            library=library,
            cxx=args.cxx,
        )
        raw_report = root / "report.json"
        subprocess.run(
            command(
                str(binary),
                str(args.model),
                str(sidecar),
                str(raw_report),
                args,
                [str(path) for path in getattr(args, "custom_op", ())],
            ),
            check=True,
        )
        report = json.loads(raw_report.read_text(encoding="utf-8"))
    return finalize(
        report,
        model=args.model,
        sidecar_summary=summary,
        output=output,
        remote=None,
        custom_ops=getattr(args, "custom_op", ()),
        platform=args.platform,
        backend=args.backend,
        header=header,
        ort_runtime={"path": str(library.resolve())},
        measured_graph_variant=(
            args.graph_variant
            if hasattr(args, "graph_variant")
            else validate_graph_variant(args.platform, args.backend, args.model)
        ),
    )


def _ensure_remote_runtime(
    host: str, options: list[str], remote_dir: str, python: str
) -> str:
    probe = _ssh(
        host,
        options,
        [python, "-c", "import onnxruntime"],
        check=False,
        capture=True,
    )
    if probe.returncode == 0:
        return python
    environment = f"{remote_dir}/venv"
    uv_probe = _ssh(
        host,
        options,
        [
            "sh", "-lc",
            'command -v uv || { test -x "$HOME/.local/bin/uv" && printf "%s\\n" "$HOME/.local/bin/uv"; }',
        ],
        check=False,
        capture=True,
    )
    remote_uv = uv_probe.stdout.strip()
    if remote_uv:
        _ssh(host, options, [remote_uv, "venv", "--python", "3.10", environment])
        python = f"{environment}/bin/python"
        _ssh(
            host,
            options,
            [remote_uv, "pip", "install", "--python", python, f"onnxruntime=={ORT_VERSION}"],
        )
    else:
        _ssh(host, options, [python, "-m", "venv", environment])
        python = f"{environment}/bin/python"
        _ssh(host, options, [python, "-m", "pip", "install", f"onnxruntime=={ORT_VERSION}"])
    return python


def benchmark_remote(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    host = args.remote
    options = args.ssh_option
    created = _ssh(
        host,
        options,
        ["mktemp", "-d", "/tmp/fiber-deployment.XXXXXX"],
        capture=True,
    )
    remote_dir = created.stdout.strip()
    if not remote_dir.startswith("/tmp/fiber-deployment."):
        raise RuntimeError(f"unexpected remote temporary path: {remote_dir!r}")
    try:
        with tempfile.TemporaryDirectory(prefix="fiber-deployment-stage.") as temporary:
            local_root = Path(temporary)
            sidecar = local_root / "initializers.bin"
            summary = write_initializer_sidecar(args.model, sidecar)
            header = ensure_ort_header()
            remote_model = f"{remote_dir}/model.onnx"
            remote_sidecar = f"{remote_dir}/initializers.bin"
            remote_source = f"{remote_dir}/benchmark_deployment.cpp"
            remote_header = f"{remote_dir}/onnxruntime_c_api.h"
            remote_binary = f"{remote_dir}/benchmark_deployment"
            remote_report = f"{remote_dir}/report.json"
            _scp(str(args.model), f"{host}:{remote_model}", options)
            _scp(str(sidecar), f"{host}:{remote_sidecar}", options)
            _scp(str(SOURCE), f"{host}:{remote_source}", options)
            _scp(str(header), f"{host}:{remote_header}", options)
            remote_custom_ops = []
            for index, library_path in enumerate(args.custom_op):
                remote_library = f"{remote_dir}/custom-op-{index}.so"
                _scp(str(library_path), f"{host}:{remote_library}", options)
                remote_custom_ops.append(remote_library)

            local_ort_library = getattr(args, "ort_library", None)
            remote_ort_library = getattr(args, "remote_ort_library", None)
            if local_ort_library is not None:
                library = f"{remote_dir}/libonnxruntime.so.1.19.2"
                _scp(str(local_ort_library), f"{host}:{library}", options)
                ort_runtime = {
                    "path": str(local_ort_library.resolve()),
                    "remote_runtime_path": library,
                }
            elif remote_ort_library:
                library = remote_ort_library
                ort_runtime = {"path": library}
            else:
                remote_python = _ensure_remote_runtime(
                    host, options, remote_dir, args.remote_python
                )
                library_probe = _ssh(
                    host,
                    options,
                    [
                        remote_python,
                        "-c",
                        "import pathlib,onnxruntime; p=pathlib.Path(onnxruntime.__file__).parent/'capi'; print(next(p.glob('libonnxruntime.so.*')))",
                    ],
                    capture=True,
                )
                library = library_probe.stdout.strip()
                ort_runtime = {
                    "path": library,
                    "python": remote_python,
                }
            if not library.startswith("/"):
                raise RuntimeError(f"unexpected remote ORT library path: {library!r}")
            runtime_link = f"{remote_dir}/libonnxruntime.so.1"
            if library != runtime_link:
                _ssh(host, options, ["ln", "-s", library, runtime_link])
            _ssh(
                host,
                options,
                [
                    args.cxx,
                    "-O3", "-DNDEBUG", "-std=c++17", "-Wall", "-Wextra", "-pedantic",
                    remote_source,
                    f"-I{remote_dir}",
                    library,
                    "-Wl,-rpath,$ORIGIN",
                    "-o", remote_binary,
                ],
            )
            _ssh(
                host,
                options,
                command(
                    remote_binary,
                    remote_model,
                    remote_sidecar,
                    remote_report,
                    args,
                    remote_custom_ops,
                ),
            )
            local_raw = local_root / "report.json"
            _scp(f"{host}:{remote_report}", str(local_raw), options)
            report = json.loads(local_raw.read_text(encoding="utf-8"))
        return finalize(
            report,
            model=args.model,
            sidecar_summary=summary,
            output=output,
            remote=host,
            custom_ops=args.custom_op,
            platform=args.platform,
            backend=args.backend,
            header=header,
            ort_runtime=ort_runtime,
            measured_graph_variant=(
                args.graph_variant
                if hasattr(args, "graph_variant")
                else validate_graph_variant(args.platform, args.backend, args.model)
            ),
        )
    finally:
        if not args.keep_remote:
            _ssh(host, options, ["rm", "-rf", remote_dir], check=False)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--platform", required=True, choices=tuple(PLATFORM_CANDIDATES))
    result.add_argument("--backend", required=True)
    result.add_argument("--model", type=Path, required=True)
    result.add_argument("--output", type=Path)
    result.add_argument("--cpu", default="auto")
    result.add_argument("--warmup", type=int, default=500)
    result.add_argument("--iterations", type=int, default=5000)
    result.add_argument("--repeats", type=int, default=5)
    result.add_argument("--seed", type=int, default=1)
    result.add_argument("--frame-ms", type=float, default=16.0)
    result.add_argument("--remote")
    result.add_argument("--remote-python", default="python3")
    result.add_argument("--runtime-python", default=str(Path(".venv/bin/python").resolve()))
    result.add_argument(
        "--ort-library",
        type=Path,
        help="local ORT 1.19.2 shared library; copied to the target for remote runs",
    )
    result.add_argument(
        "--remote-ort-library",
        help="existing absolute ORT 1.19.2 shared-library path on the remote target",
    )
    result.add_argument("--cxx", default="g++")
    result.add_argument("--ssh-option", action="append", default=[])
    result.add_argument("--keep-remote", action="store_true")
    result.add_argument("--ort-info", action="store_true")
    result.add_argument(
        "--embedded-initializers",
        action="store_true",
        help="let ORT own embedded initializers (needed by some QLinear kernels)",
    )
    result.add_argument(
        "--custom-op",
        type=Path,
        action="append",
        default=[],
        help="optimized runtime library; repeat when a candidate has multiple components",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    args.model = args.model.resolve()
    args.custom_op = [path.resolve() for path in args.custom_op]
    if args.ort_library is not None:
        args.ort_library = args.ort_library.resolve()
    if args.ort_library is not None and args.remote_ort_library:
        raise SystemExit("choose only one of --ort-library and --remote-ort-library")
    try:
        validate_candidate(args.platform, args.backend, args.custom_op)
        args.graph_variant = validate_graph_variant(args.platform, args.backend, args.model)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    output = (
        args.output.resolve()
        if args.output
        else args.model.with_suffix(f".{args.backend.lower()}.benchmark.json")
    )
    report = benchmark_remote(args, output) if args.remote else benchmark_local(args, output)
    print(json.dumps(report["latency"], indent=2), flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import random
import statistics
import tempfile
from pathlib import Path
from typing import Any

from deployment.benchmark import FP32_PROTOCOL, GRAPH_VARIANTS, SOURCE, _scp, _ssh
from deployment.support import ensure_ort_header, write_initializer_sidecar


MODELS = (
    "fiber_c",
    "fiber_b",
    "fiber_e",
    "fastenhancer_t",
    "fastenhancer_b",
    "fastenhancer_s",
)
PLATFORM = "AMD-Ryzen"
FP32_BACKEND = FP32_PROTOCOL[PLATFORM]["backend"]
BACKENDS = (FP32_BACKEND, "VNNI-W8A8")
RUNTIME_NAMES = {FP32_BACKEND: "fp32.so", "VNNI-W8A8": "vnni-w8a8.so"}


def _run(
    host: str,
    options: list[str],
    root: str,
    runner: str,
    model: str,
    backend: str,
    output: str,
    args: argparse.Namespace,
    *,
    warmup: int | None = None,
    iterations: int | None = None,
    repeats: int | None = None,
) -> None:
    variant = (
        "native" if GRAPH_VARIANTS[PLATFORM][backend] == "native-temporal" else "fused"
    )
    runtime = RUNTIME_NAMES[backend]
    command = [
        "env",
        f"LD_LIBRARY_PATH={root}",
        runner,
        "--model", f"{root}/graphs/{variant}/{model}.onnx",
        "--initializers", f"{root}/sidecars/{variant}/{model}.bin",
        "--custom-op", f"{root}/runtimes/{runtime}",
        "--output", output,
        "--cpu", str(args.cpu),
        "--warmup", str(args.warmup if warmup is None else warmup),
        "--iterations", str(args.iterations if iterations is None else iterations),
        "--repeats", str(args.repeats if repeats is None else repeats),
        "--seed", str(args.seed),
        "--frame-ms", "16",
    ]
    _ssh(host, options, command)


def run(args: argparse.Namespace) -> dict[str, Any]:
    models = tuple(args.model) if getattr(args, "model", None) else MODELS
    graphs = args.graphs.resolve()
    runtimes = {
        "fp32.so": args.runtime_fp32.resolve(),
        "vnni-w8a8.so": args.runtime_w8a8.resolve(),
    }
    ort_library = args.ort_library.resolve()
    host, options = args.remote, args.ssh_option
    created = _ssh(
        host,
        options,
        ["mktemp", "-d", "/tmp/fiber-paired-benchmark.XXXXXX"],
        capture=True,
    )
    remote_root = created.stdout.strip()
    if not remote_root.startswith("/tmp/fiber-paired-benchmark."):
        raise RuntimeError(f"unexpected remote temporary path: {remote_root!r}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        with tempfile.TemporaryDirectory(prefix="fiber-paired-stage.") as temporary:
            stage = Path(temporary)
            header = ensure_ort_header()
            for directory in ("graphs/native", "graphs/fused", "sidecars/native", "sidecars/fused", "runtimes", "reports"):
                _ssh(host, options, ["mkdir", "-p", f"{remote_root}/{directory}"])
            _scp(str(SOURCE), f"{host}:{remote_root}/runner.cc", options)
            _scp(str(header), f"{host}:{remote_root}/onnxruntime_c_api.h", options)
            _scp(str(ort_library), f"{host}:{remote_root}/libonnxruntime.so.1.19.2", options)
            for name, runtime in runtimes.items():
                _scp(str(runtime), f"{host}:{remote_root}/runtimes/{name}", options)
            graph_rows = {}
            for variant in ("native", "fused"):
                for model in models:
                    graph = graphs / f"{model}.{variant}.onnx"
                    sidecar = stage / f"{variant}-{model}.bin"
                    sidecar_report = write_initializer_sidecar(graph, sidecar)
                    _scp(str(graph), f"{host}:{remote_root}/graphs/{variant}/{model}.onnx", options)
                    _scp(str(sidecar), f"{host}:{remote_root}/sidecars/{variant}/{model}.bin", options)
                    graph_rows[f"{model}.{variant}"] = {
                        "onnx": str(graph),
                        "initializer_count": sidecar_report["initializer_count"],
                        "initializer_bytes": sidecar_report["tensor_bytes"],
                    }
            _ssh(
                host,
                options,
                [
                    "ln", "-s", f"{remote_root}/libonnxruntime.so.1.19.2",
                    f"{remote_root}/libonnxruntime.so.1",
                ],
            )
            runner = f"{remote_root}/runner"
            _ssh(
                host,
                options,
                [
                    args.cxx,
                    "-O3", "-DNDEBUG", "-std=c++17", "-Wall", "-Wextra", "-pedantic",
                    f"{remote_root}/runner.cc", f"-I{remote_root}",
                    f"{remote_root}/libonnxruntime.so.1.19.2", "-Wl,-rpath,$ORIGIN",
                    "-o", runner,
                ],
            )
            precondition = f"{remote_root}/reports/precondition.json"
            _run(
                host,
                options,
                remote_root,
                runner,
                "fiber_e" if "fiber_e" in models else models[0],
                "VNNI-W8A8",
                precondition,
                args,
                warmup=0,
                iterations=args.precondition_frames,
                repeats=1,
            )
            rng = random.Random(args.seed ^ 0xA7D)
            initial_backend = {model: rng.randrange(2) for model in models}
            order = []
            report_paths: dict[str, list[Path]] = {
                f"{model}.{backend}": []
                for model in models
                for backend in BACKENDS
            }
            for round_index in range(args.rounds):
                round_models = list(models)
                rng.shuffle(round_models)
                for model in round_models:
                    backends = list(BACKENDS)
                    if (round_index + initial_backend[model]) % 2:
                        backends.reverse()
                    for backend in backends:
                        filename = f"round-{round_index + 1:02d}.{model}.{backend.lower()}.json"
                        remote_report = f"{remote_root}/reports/{filename}"
                        print(f"[{round_index + 1}/{args.rounds}] {model} {backend}", flush=True)
                        _run(
                            host,
                            options,
                            remote_root,
                            runner,
                            model,
                            backend,
                            remote_report,
                            args,
                        )
                        local_report = output / filename
                        _scp(f"{host}:{remote_report}", str(local_report), options)
                        report_paths[f"{model}.{backend}"].append(local_report)
                        order.append(
                            {
                                "round": round_index + 1,
                                "model": model,
                                "backend": backend,
                                "report": filename,
                            }
                        )
            local_precondition = output / "precondition.json"
            _scp(f"{host}:{precondition}", str(local_precondition), options)

        summaries = {}
        for key, paths in report_paths.items():
            repeat_means = [
                float(row["mean_ms"])
                for path in paths
                for row in json.loads(path.read_text(encoding="utf-8"))["latency"]["repeats"]
            ]
            summaries[key] = {
                "median_repeat_mean_ms": statistics.median(repeat_means),
                "mean_repeat_mean_ms": statistics.fmean(repeat_means),
                "sample_std_ms": statistics.stdev(repeat_means),
                "repeat_means_ms": repeat_means,
            }
        return {
            "schema": "fiber-se.ryzen-balanced-confirmation.v5",
            "protocol": {
                "precondition_frames": args.precondition_frames,
                "rounds": args.rounds,
                "repeats_per_round": args.repeats,
                "warmup_per_repeat": args.warmup,
                "iterations_per_repeat": args.iterations,
                "fp32_backend": FP32_BACKEND,
                "fp32_graph_variant": FP32_PROTOCOL[PLATFORM]["graph_variant"],
                "selective_w8a8_backend": "VNNI-W8A8",
                "within_model_backend_order": "alternating",
                "model_order": "seeded-random",
                "seed": args.seed,
                "primary_statistic": "median of balanced repeat means",
            },
            "artifacts": {
                "graphs": graph_rows,
                "runtimes": {name: str(path) for name, path in runtimes.items()},
                "runner_source": str(SOURCE),
                "onnxruntime_c_api_header": str(header),
                "onnxruntime": str(ort_library),
            },
            "order": order,
            "results": summaries,
        }
    finally:
        if not args.keep_remote:
            _ssh(host, options, ["rm", "-rf", remote_root], check=False)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--remote", required=True)
    result.add_argument(
        "--model",
        action="append",
        default=[],
        help="model basename to include; repeat for a custom suite (defaults to six release points)",
    )
    result.add_argument("--graphs", type=Path, required=True)
    result.add_argument("--runtime-fp32", type=Path, required=True)
    result.add_argument("--runtime-w8a8", type=Path, required=True)
    result.add_argument("--ort-library", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--cpu", type=int, default=4)
    result.add_argument("--rounds", type=int, default=8)
    result.add_argument("--repeats", type=int, default=1)
    result.add_argument("--warmup", type=int, default=10_000)
    result.add_argument("--iterations", type=int, default=10_000)
    result.add_argument("--precondition-frames", type=int, default=200_000)
    result.add_argument("--seed", type=int, default=20260821)
    result.add_argument("--cxx", default="g++")
    result.add_argument("--ssh-option", action="append", default=[])
    result.add_argument("--keep-remote", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    report = run(args)
    destination = args.output.resolve() / "manifest.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["results"], indent=2), flush=True)


if __name__ == "__main__":
    main()

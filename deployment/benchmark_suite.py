from __future__ import annotations

import argparse
import json
import random
import re
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

from deployment.benchmark import (
    FP32_PROTOCOL,
    GRAPH_VARIANTS,
    PLATFORM_CANDIDATES,
    benchmark_remote,
    validate_candidate,
    validate_graph_variant,
)


POINTS = (
    "fiber_c",
    "fiber_b",
    "fiber_e",
    "fastenhancer_t",
    "fastenhancer_b",
    "fastenhancer_s",
)
DEFAULT_REPEATS = {"Xeon": 7, "AmpereOne": 10, "AMD-Ryzen": 1, "RDK-X5": 5}
DEFAULT_SEEDS = {
    "Xeon": 20260901,
    "AmpereOne": 20260901,
    "AMD-Ryzen": 20260901,
    "RDK-X5": 20260901,
}
# AMD uses median for stability, for laptop env is unstable

def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def key_values(values: list[str], option: str) -> dict[str, str]:
    result = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key or not item:
            raise SystemExit(f"{option} expects NAME=VALUE, got {value!r}")
        result[key] = item
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    points = tuple(args.point) if getattr(args, "point", None) else POINTS
    candidates = PLATFORM_CANDIDATES[args.platform]
    runtimes = key_values(args.runtime, "--runtime")
    missing = set(candidates) - set(runtimes)
    if missing:
        raise SystemExit(f"missing --runtime entries: {', '.join(sorted(missing))}")
    graph_root = args.graphs.resolve()
    output = args.output.resolve()
    candidate_root = output / "candidates"
    candidate_root.mkdir(parents=True, exist_ok=True)
    repeats = args.repeats or DEFAULT_REPEATS[args.platform]
    seed = args.seed if args.seed is not None else DEFAULT_SEEDS[args.platform]

    reports: dict[str, dict[str, Any]] = {point: {} for point in points}
    schedule = [(point, backend) for point in points for backend in candidates]
    random.Random(seed ^ 0xB553).shuffle(schedule)
    for point, backend in schedule:
        variant = GRAPH_VARIANTS[args.platform][backend]
        suffix = "native" if variant == "native-temporal" else "fused"
        model = graph_root / f"{point}.{suffix}.onnx"
        runtime = Path(runtimes[backend]).expanduser().resolve()
        report_path = candidate_root / f"{point}.{slug(backend)}.json"
        if not model.is_file() or not runtime.is_file():
            raise FileNotFoundError(model if not model.is_file() else runtime)
        validate_candidate(args.platform, backend, [runtime])
        validate_graph_variant(args.platform, backend, model)
        if not report_path.is_file():
            print(f"[{args.platform}] {point} / {backend}", flush=True)
            benchmark_remote(
                Namespace(
                    platform=args.platform,
                    backend=backend,
                    model=model,
                    output=report_path,
                    cpu=args.cpu,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    repeats=repeats,
                    seed=seed,
                    frame_ms=16.0,
                    remote=args.remote,
                    remote_python=args.remote_python,
                    ort_library=None,
                    remote_ort_library=args.remote_ort_library,
                    cxx=args.cxx,
                    ssh_option=args.ssh_option,
                    keep_remote=False,
                    ort_info=True,
                    custom_op=[runtime],
                    runtime_python=sys.executable,
                    graph_variant=variant,
                ),
                report_path,
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["platform"] != args.platform or report["backend"] != backend:
            raise RuntimeError(f"stale candidate report: {report_path}")
        reports[point][backend] = {
            "mean_ms": float(report["latency"]["mean_ms"]),
            "p50_ms": float(report["latency"]["p50_ms"]),
            "p90_ms": float(report["latency"]["p90_ms"]),
            "p99_ms": float(report["latency"]["p99_ms"]),
            "repeat_mean_sample_std_ms": float(
                report["latency"]["repeat_mean_sample_std_ms"]
            ),
            "graph_variant": variant,
            "report": str(report_path.resolve()),
            "onnx": str(model),
            "runtime": str(runtime),
        }

    selected = {}
    for point, rows in reports.items():
        backend = min(candidates, key=lambda name: rows[name]["mean_ms"])
        selected[point] = {"backend": backend, **rows[backend]}
    first_report = json.loads(
        Path(reports[points[0]][candidates[0]]["report"]).read_text(encoding="utf-8")
    )
    manifest = {
        "schema": "fiber-se.formal-platform-suite.v4",
        "platform": args.platform,
        "hardware": {
            "host": first_report["host"],
            "cpu_model": first_report["cpu_model"],
            "machine": first_report["machine"],
            "selected_cpu": first_report["selected_cpu"],
        },
        "protocol": {
            "candidate_set": list(candidates),
            "fp32": {
                **FP32_PROTOCOL[args.platform],
                "selection": "fixed for this platform",
            },
            "selective_w8a8_candidates": [
                backend for backend in candidates if "W8A8" in backend
            ],
            "selection_rule": "minimum formal all-frame mean per static model shape",
            "family_tag_used": False,
            "threads": 1,
            "continuous_state": True,
            "io_binding": True,
            "shared_initializer_bank_cache": True,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": repeats,
            "seed": seed,
            "execution_order_rule": "seeded-random over model/backend pairs",
            "execution_order": [
                {"point": point, "backend": backend} for point, backend in schedule
            ],
            "frame_ms": 16.0,
        },
        "candidates": reports,
        "selected": selected,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--platform", required=True, choices=tuple(PLATFORM_CANDIDATES))
    result.add_argument(
        "--point",
        action="append",
        default=[],
        help="model basename to include; repeat for a custom suite (defaults to six release points)",
    )
    result.add_argument("--graphs", type=Path, required=True)
    result.add_argument("--runtime", action="append", default=[], metavar="BACKEND=PATH")
    result.add_argument("--remote", required=True)
    result.add_argument("--remote-ort-library", required=True)
    result.add_argument("--remote-python", default="python3")
    result.add_argument("--cpu", default="auto")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--warmup", type=int, default=10_000)
    result.add_argument("--iterations", type=int, default=10_000)
    result.add_argument("--repeats", type=int)
    result.add_argument("--seed", type=int)
    result.add_argument("--cxx", default="g++")
    result.add_argument("--ssh-option", action="append", default=[])
    return result


def main() -> None:
    manifest = run(parser().parse_args())
    print(json.dumps(manifest["selected"], indent=2), flush=True)


if __name__ == "__main__":
    main()

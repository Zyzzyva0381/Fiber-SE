import argparse
import subprocess
import sys
from pathlib import Path


POINTS = (
    "fastenhancer_t",
    "fastenhancer_b",
    "fastenhancer_s",
    "fiber_c",
    "fiber_b",
    "fiber_e",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("build/graphs"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for point in POINTS:
        raw = args.output / f"{point}.onnx"
        native = args.output / f"{point}.native.onnx"
        fused = args.output / f"{point}.fused.onnx"
        w8 = args.output / f"{point}.w8.onnx"
        subprocess.run(
            [
                sys.executable,
                "export_onnx.py",
                "--config",
                f"configs/{point}.yaml",
                "--checkpoint",
                str(args.checkpoints / f"{point}.pt"),
                "--output",
                str(raw),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "stream_aot.py",
                "--input",
                str(raw),
                "--output",
                str(native),
                "--complex-pipeline",
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "stream_aot.py",
                "--input",
                str(raw),
                "--output",
                str(fused),
                "--complex-pipeline",
                "--temporal-blocks",
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "quantize.py",
                str(raw),
                str(w8),
                "--report",
                str(w8.with_suffix(".json")),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()

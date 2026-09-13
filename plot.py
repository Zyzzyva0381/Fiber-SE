import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


POINTS = (
    "fastenhancer_t",
    "fastenhancer_b",
    "fastenhancer_s",
    "fiber_c",
    "fiber_b",
    "fiber_e",
)
FAMILIES = {
    "FastEnhancer": POINTS[:3],
    "Fiber": POINTS[3:],
}
LABELS = {
    "fastenhancer_t": "FE-T",
    "fastenhancer_b": "FE-B",
    "fastenhancer_s": "FE-S",
    "fiber_c": "Fiber-C",
    "fiber_b": "Fiber-B",
    "fiber_e": "Fiber-E",
}
HARDWARE = {
    "xeon": "Intel Xeon Silver 4410Y",
    "ampereone": "AmpereOne A192-32X",
    "ryzen": "AMD Ryzen 7 7840H",
    "rdk-x5": "RDK X5 Cortex-A55",
}
TEST = ("SISNR", "SDR", "PESQ", "ESTOI", "STOI", "OVRL", "SIG", "BAK", "P808_MOS")
BLIND = ("OVRL", "SIG", "BAK", "P808_MOS")


def quality(path: Path, split: str) -> dict[str, float]:
    aggregate = json.loads(path.read_text())["aggregate"]
    if split == "test1000":
        aggregate = {**aggregate["intrusive"], **aggregate["dnsmos"]}
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    for platform, hardware in HARDWARE.items():
        manifest = json.loads((args.results / f"latency/{platform}/manifest.json").read_text())
        if "selected" in manifest:
            rows = manifest["selected"]
        else:
            rows = {}
            for point in POINTS:
                candidates = {
                    backend: manifest["results"][f"{point}.{backend}"]["median_repeat_mean_ms"]
                    for backend in ("FP32", "VNNI-W8A8")
                }
                backend = min(candidates, key=candidates.get)
                rows[point] = {"backend": backend, "mean_ms": candidates[backend]}
        for split, metrics, shape in (("test1000", TEST, (3, 3)), ("blind600", BLIND, (2, 2))):
            scores = {}
            for point in POINTS:
                precision = "w8a8" if "W8A8" in rows[point]["backend"].upper() else "fp32"
                scores[point] = quality(
                    args.results / f"quality/{precision}/{split}/{point}.json", split
                )
            figure, axes = plt.subplots(*shape, figsize=(14.2, 12.2 if shape[0] == 3 else 8.2))
            for axis, metric in zip(axes.flat, metrics):
                for family, points in FAMILIES.items():
                    ordered = sorted(points, key=lambda point: rows[point]["mean_ms"])
                    axis.plot(
                        [rows[point]["mean_ms"] for point in ordered],
                        [scores[point][metric] for point in ordered],
                        marker="o" if family == "Fiber" else "D",
                        linestyle="-" if family == "Fiber" else "--",
                        label=family,
                    )
                    for point in ordered:
                        axis.annotate(
                            LABELS[point],
                            (rows[point]["mean_ms"], scores[point][metric]),
                            xytext=(4, 5),
                            textcoords="offset points",
                            fontsize=7,
                        )
                axis.set_title(metric)
                axis.set_xlabel("Streaming latency (ms / 16 ms frame)")
                axis.grid(alpha=0.3)
            figure.suptitle(f"DNS3 {split}: Fiber vs matched FastEnhancer on {hardware}")
            handles, labels = axes.flat[0].get_legend_handles_labels()
            figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
            figure.tight_layout(rect=(0, 0, 1, 0.95))
            output = args.results / f"figures/{platform}_{split}.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output, dpi=190, bbox_inches="tight")
            figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
            plt.close(figure)


if __name__ == "__main__":
    main()

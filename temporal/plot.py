import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


P = (1, 2, 4, 8, 16)
PLATFORMS = ("xeon", "ampereone", "ryzen", "rdk-x5")
LABELS = ("Xeon", "AmpereOne", "Ryzen", "RDK-X5")
FIELDS = ("w8_quantize_us", "w8_dot_us", "w8_full_us")
TITLES = ("(a) Activation scan/pack", "(b) INT8 GEMM + dequant.", "(c) Total temporal affine")
STYLES = (("o", "-"), ("s", "--"), ("^", ":"), ("D", "-."))
COLORS = ("#0072B2", "#D55E00", "#009E73", "#8A63C7")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values = {}
    for platform in PLATFORMS:
        rows = [json.loads(line) for line in (args.input / f"{platform}.jsonl").read_text().splitlines()]
        values[platform] = {
            field: [
                statistics.median(float(row[field]) for row in rows if row["p"] == p)
                for p in P
            ]
            for field in FIELDS
        }
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    for axis, field, title in zip(axes, FIELDS, TITLES):
        for platform, label, style, color in zip(PLATFORMS, LABELS, STYLES, COLORS):
            row = values[platform][field]
            normalized = [value / row[0] for value in row]
            axis.plot(
                P,
                normalized,
                marker=style[0],
                linestyle=style[1],
                color=color,
                label=label,
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks(P, labels=P)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.3)
        axis.axhline(1.0, color="black", linewidth=0.7, linestyle="--")
    axes[0].set_ylabel("normalised time (p=1)")
    axes[0].set_yscale("log", base=2)
    axes[0].set_yticks((1, 0.5, 0.25, 0.125), labels=("1", "1/2", "1/4", "1/8"))
    axes[2].axvline(4, color="black", linewidth=0.8, linestyle=":")
    axes[2].text(4.4, 2.45, r"$p^{*}=4$")
    figure.supxlabel("grouping factor p")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0.04, 1, 0.9))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")


if __name__ == "__main__":
    main()

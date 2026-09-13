import argparse
import json
import math
from pathlib import Path

import soundfile as sf

from deployment.evaluate import StreamingDeployment
from evaluation.metrics import DNSMOSMetrics


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--custom-op", type=Path, action="append", default=[])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    deployment = StreamingDeployment(args.model, args.custom_op)
    metric = DNSMOSMetrics(
        root / "DNSMOS/sig_bak_ovr.onnx", root / "DNSMOS/model_v8.onnx"
    )
    rows = []
    for index, path in enumerate(sorted(args.input.glob("*.wav"))):
        waveform, rate = sf.read(path, dtype="float32")
        enhanced = deployment(waveform)
        values = metric(enhanced, rate)
        rows.append({"index": index, "uid": path.stem, **values})
    metrics = ("OVRL", "SIG", "BAK", "P808_MOS")
    report = {
        "items": len(rows),
        "uid_count": len(rows),
        "unique_uid_count": len({row["uid"] for row in rows}),
        "deployment": {
            "onnx": {"path": str(args.model.resolve())},
            "custom_ops": [str(path.resolve()) for path in args.custom_op],
        },
        "aggregate": {
            name: math.fsum(float(row[name]) for row in rows) / len(rows)
            for name in metrics
        },
        "utterances": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

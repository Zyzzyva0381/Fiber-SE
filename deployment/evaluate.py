from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import platform
import socket
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

from evaluation.metrics import DNSMOSMetrics, intrusive_metrics


INTRUSIVE = ("SISNR", "SDR", "PESQ", "ESTOI", "STOI")
DNSMOS = ("OVRL", "SIG", "BAK", "P808_MOS")
_DNSMOS: DNSMOSMetrics | None = None


def _initialize_worker(primary: str, p808: str) -> None:
    global _DNSMOS
    _DNSMOS = DNSMOSMetrics(Path(primary), Path(p808))


def _score(payload: tuple[int, str, np.ndarray, np.ndarray]) -> dict[str, Any]:
    if _DNSMOS is None:
        raise RuntimeError("DNSMOS worker was not initialized")
    index, uid, clean, enhanced = payload
    intrusive = intrusive_metrics(clean, enhanced, 16000, enabled=INTRUSIVE)
    dnsmos = _DNSMOS(enhanced, 16000)
    values = (*intrusive.values(), *dnsmos.values())
    if not all(math.isfinite(float(value)) for value in values):
        raise RuntimeError(f"non-finite metric for {uid}")
    return {
        "index": index,
        "uid": uid,
        "intrusive": {key: float(intrusive[key]) for key in INTRUSIVE},
        "dnsmos": {key: float(dnsmos[key]) for key in DNSMOS},
    }


def _session(model: Path, custom_ops: list[Path]) -> ort.InferenceSession:
    options = ort.SessionOptions()
    for library in custom_ops:
        options.register_custom_ops_library(str(library))
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])


class StreamingDeployment:
    def __init__(self, model: Path, custom_ops: list[Path]) -> None:
        self.session = _session(model, custom_ops)
        self.inputs = [value.name for value in self.session.get_inputs()]
        self.outputs = [value.name for value in self.session.get_outputs()]
        self.state_shapes = [
            tuple(int(dimension) for dimension in value.shape)
            for value in self.session.get_inputs()[1:]
        ]

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(waveform).unsqueeze(0)
        window = torch.hann_window(512, dtype=torch.float32)
        spectrum = torch.stft(
            tensor,
            512,
            hop_length=256,
            win_length=512,
            window=window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        frames = torch.view_as_real(spectrum).numpy()
        states = [np.zeros(shape, dtype=np.float32) for shape in self.state_shapes]
        enhanced = []
        for index in range(frames.shape[2]):
            feeds = {self.inputs[0]: frames[:, :, index : index + 1, :]}
            feeds.update(dict(zip(self.inputs[1:], states)))
            values = self.session.run(self.outputs, feeds)
            enhanced.append(np.asarray(values[0], dtype=np.float32))
            states = [np.asarray(value, dtype=np.float32) for value in values[1:]]
        reconstructed = torch.istft(
            torch.view_as_complex(
                torch.from_numpy(np.ascontiguousarray(np.concatenate(enhanced, axis=2)))
            ),
            512,
            hop_length=256,
            win_length=512,
            window=window,
            center=True,
            normalized=False,
            onesided=True,
            length=waveform.size,
        )
        return np.asarray(reconstructed[0], dtype=np.float32)


def _ort_library() -> Path:
    root = Path(ort.__file__).resolve().parent / "capi"
    return next(root.glob("libonnxruntime.so.*"))


def run(args: argparse.Namespace) -> dict[str, Any]:
    model = args.model.resolve()
    custom_ops = [path.resolve() for path in args.custom_op]
    primary = args.primary_model.resolve()
    p808 = args.p808_model.resolve()
    pairs = []
    clean = {path.stem: path for path in args.clean_dir.resolve().glob("*.wav")}
    for noisy in sorted(args.noisy_dir.resolve().glob("*.wav")):
        if noisy.stem not in clean:
            raise RuntimeError(f"missing clean pair for {noisy.stem}")
        pairs.append((noisy, clean[noisy.stem]))
    if len(pairs) != args.expected_items:
        raise RuntimeError(f"expected {args.expected_items} pairs, found {len(pairs)}")

    enhanced_dir = args.enhanced_dir.resolve()
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    deployment = StreamingDeployment(model, custom_ops)
    context = multiprocessing.get_context("spawn")
    pending: deque[Any] = deque()
    rows = []
    outputs = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=_initialize_worker,
        initargs=(str(primary), str(p808)),
    ) as pool:
        for index, (noisy_path, clean_path) in enumerate(pairs):
            noisy, noisy_rate = sf.read(noisy_path, dtype="float32", always_2d=False)
            reference, clean_rate = sf.read(clean_path, dtype="float32", always_2d=False)
            if noisy_rate != 16000 or clean_rate != 16000 or noisy.shape != reference.shape:
                raise RuntimeError(f"invalid pair {noisy_path.stem}")
            prediction = deployment(np.asarray(noisy, dtype=np.float32))
            output = enhanced_dir / noisy_path.name
            sf.write(output, prediction, 16000, subtype="FLOAT")
            scored, rate = sf.read(output, dtype="float32", always_2d=False)
            if rate != 16000:
                raise RuntimeError(f"invalid enhanced output {output}")
            outputs.append(
                {
                    "index": index,
                    "uid": noisy_path.stem,
                    "samples": int(scored.size),
                }
            )
            pending.append(pool.submit(_score, (index, noisy_path.stem, reference, scored)))
            if len(pending) >= 2 * args.workers:
                rows.append(pending.popleft().result())
            if (index + 1) % 25 == 0:
                print(json.dumps({"inferred": index + 1, "total": len(pairs)}), flush=True)
        while pending:
            rows.append(pending.popleft().result())
    rows.sort(key=lambda row: row["index"])
    uids = [path.stem for path, _ in pairs]
    aggregate = {
        group: {
            metric: math.fsum(row[group][metric] for row in rows) / len(rows)
            for metric in metrics
        }
        for group, metrics in (("intrusive", INTRUSIVE), ("dnsmos", DNSMOS))
    }
    library = _ort_library()
    metrics_source = Path(__file__).resolve().parents[1] / "evaluation/metrics.py"
    return {
        "schema": "fiber-se.deployment-nine-metric.v2",
        "split": "DNS3-Test1000",
        "items": len(rows),
        "uid_count": len(uids),
        "unique_uid_count": len(set(uids)),
        "host": {
            "hostname": socket.gethostname(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "deployment": {
            "onnx": {"path": str(model)},
            "custom_op_libraries": [{"path": str(path)} for path in custom_ops],
            "onnxruntime": {
                "version": ort.__version__,
                "library": str(library),
            },
            "continuous_state_within_utterance": True,
            "state_reset_between_utterances": True,
            "waveform_transport": "IEEE-float32-WAV",
            "stft_window": "Hann",
        },
        "scoring": {
            "metrics_source": {"path": str(metrics_source)},
            "dnsmos_primary": {"path": str(primary)},
            "dnsmos_p808": {"path": str(p808)},
            "workers": args.workers,
        },
        "aggregate": aggregate,
        "outputs": outputs,
        "utterances": rows,
    }


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", type=Path, required=True)
    result.add_argument("--custom-op", type=Path, action="append", default=[])
    result.add_argument("--noisy-dir", type=Path, required=True)
    result.add_argument("--clean-dir", type=Path, required=True)
    result.add_argument("--enhanced-dir", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument(
        "--primary-model",
        type=Path,
        default=root / "DNSMOS/sig_bak_ovr.onnx",
    )
    result.add_argument(
        "--p808-model",
        type=Path,
        default=root / "DNSMOS/model_v8.onnx",
    )
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--expected-items", type=int, default=1000)
    return result


def main() -> None:
    args = parser().parse_args()
    report = run(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()

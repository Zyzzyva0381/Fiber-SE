from __future__ import annotations

import argparse
import json
import math
import multiprocessing
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf

from evaluation.metrics import DNSMOSMetrics, intrusive_metrics
from dataloader import discover_pairs
from models import build_model


INTRUSIVE = ("SISNR", "SDR", "PESQ", "ESTOI", "STOI")
DNSMOS = ("OVRL", "SIG", "BAK", "P808_MOS")
_DNSMOS: DNSMOSMetrics | None = None


def _initialize_worker(primary: str, p808: str) -> None:
    global _DNSMOS
    _DNSMOS = DNSMOSMetrics(Path(primary), Path(p808))


def _score(payload: tuple[int, str, np.ndarray, np.ndarray, int]) -> dict[str, Any]:
    if _DNSMOS is None:
        raise RuntimeError("DNSMOS worker was not initialized")
    index, uid, clean, enhanced, sample_rate = payload
    intrusive = intrusive_metrics(clean, enhanced, sample_rate, enabled=INTRUSIVE)
    dnsmos = _DNSMOS(enhanced, sample_rate)
    if set(intrusive) != set(INTRUSIVE) or set(dnsmos) != set(DNSMOS):
        raise RuntimeError(f"incomplete metrics for {uid}")
    if not all(math.isfinite(float(value)) for value in (*intrusive.values(), *dnsmos.values())):
        raise RuntimeError(f"non-finite metric for {uid}")
    return {
        "index": index,
        "uid": uid,
        "intrusive": {key: float(intrusive[key]) for key in INTRUSIVE},
        "dnsmos": {key: float(dnsmos[key]) for key in DNSMOS},
    }


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    config = OmegaConf.load(config_path)
    OmegaConf.resolve(config)
    configured_model = str(config.model.name)
    selected_model = args.model_entry or configured_model
    model = build_model(selected_model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)
    model.to(device).float().eval()

    primary, p808 = args.primary_model.resolve(), args.p808_model.resolve()
    noisy_dir = (
        Path(str(config.data.test.noisy_dir)) if args.noisy_dir is None else args.noisy_dir
    ).resolve()
    clean_dir = (
        Path(str(config.data.test.clean_dir)) if args.clean_dir is None else args.clean_dir
    ).resolve()
    pairs = discover_pairs(noisy_dir, clean_dir)
    noisy_paths = [pair[0] for pair in pairs]
    clean_paths = [pair[1] for pair in pairs]
    uids = [path.stem for path in noisy_paths]
    sample_rate = int(config.data.sample_rate)

    context = multiprocessing.get_context("spawn")
    rows: list[dict[str, Any]] = []
    pending: deque[Any] = deque()
    maximum_pending = max(args.workers, args.max_pending or args.workers * 2)
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=_initialize_worker,
        initargs=(str(primary), str(p808)),
    ) as pool:
        for index, (noisy_path, clean_path) in enumerate(zip(noisy_paths, clean_paths)):
            noisy, noisy_rate = sf.read(noisy_path, dtype="float32")
            clean, clean_rate = sf.read(clean_path, dtype="float32")
            if noisy_rate != sample_rate or clean_rate != sample_rate:
                raise RuntimeError(f"sample-rate mismatch for {uids[index]}")
            if noisy.ndim != 1 or clean.ndim != 1 or noisy.shape != clean.shape:
                raise RuntimeError(f"pair shape mismatch for {uids[index]}")
            tensor = torch.from_numpy(noisy).unsqueeze(0).to(device)
            enhanced = model(tensor)
            if enhanced.shape != tensor.shape:
                raise RuntimeError(
                    f"strict output length failure for {uids[index]}: "
                    f"{tuple(tensor.shape)} != {tuple(enhanced.shape)}"
                )
            prediction = np.asarray(enhanced[0].cpu(), dtype=np.float32)
            pending.append(
                pool.submit(_score, (index, uids[index], clean, prediction, sample_rate))
            )
            if len(pending) >= maximum_pending:
                rows.append(pending.popleft().result())
            if (index + 1) % 25 == 0:
                print(json.dumps({"inferred": index + 1, "total": len(pairs)}), flush=True)
        while pending:
            rows.append(pending.popleft().result())
    rows.sort(key=lambda row: row["index"])
    if [row["uid"] for row in rows] != uids:
        raise RuntimeError("scoring results were reordered, duplicated, or lost")
    aggregate = {
        group: {
            metric: math.fsum(float(row[group][metric]) for row in rows) / len(rows)
            for metric in metrics
        }
        for group, metrics in (("intrusive", INTRUSIVE), ("dnsmos", DNSMOS))
    }
    return {
        "model": selected_model,
        "split": str(config.evaluation.split),
        "items": len(rows),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "evaluation": {
            "model_precision": "torch.float32",
            "autocast_enabled": False,
            "tf32_enabled": False,
            "waveform_transport": "in-memory-float32",
            "sample_rate_hz": sample_rate,
            "scoring_workers": args.workers,
        },
        "dnsmos_models": {
            "primary": str(primary),
            "p808": str(p808),
        },
        "aggregate": aggregate,
        "utterances": rows,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--config", type=Path, required=True)
    result.add_argument(
        "--model-entry",
        help="model module name; defaults to model.name in the config",
    )
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--noisy-dir", type=Path)
    result.add_argument("--clean-dir", type=Path)
    result.add_argument(
        "--primary-model",
        type=Path,
        default=Path(__file__).parent / "DNSMOS" / "sig_bak_ovr.onnx",
    )
    result.add_argument(
        "--p808-model",
        type=Path,
        default=Path(__file__).parent / "DNSMOS" / "model_v8.onnx",
    )
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--max-pending", type=int, default=0)
    return result


def main() -> None:
    args = parser().parse_args()
    output = args.output.resolve()
    report = evaluate(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()

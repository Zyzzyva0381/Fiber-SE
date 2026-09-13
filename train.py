from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from dataloader import (
    DNS3SegmentDataset,
    EpochSubsetSampler,
    discover_pairs,
    seed_worker,
)
from evaluation.metrics import intrusive_metrics
from loss_factory import DNS3Loss, build_optimizer
from models import build_model
from scheduler import CosineAnnealingWarmup


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_save(payload: Any, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".{uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingWarmup,
    scaler: torch.amp.GradScaler,
    epoch: int,
    history: list[dict[str, Any]],
    best_pesq: float,
    best_epoch: int,
) -> dict[str, Any]:
    return {
        "schema": "fiber-se.checkpoint.v1",
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "history": history,
        "best_dev_pesq": float(best_pesq),
        "best_dev_epoch": int(best_epoch),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }


def _restore_rng(payload: dict[str, Any]) -> None:
    state = payload["rng"]
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _mean(rows: list[dict[str, float]], weights: list[int] | None = None) -> dict[str, float]:
    if not rows:
        return {}
    if weights is None:
        return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    total = float(sum(weights))
    return {
        key: float(sum(row[key] * weight for row, weight in zip(rows, weights)) / total)
        for key in rows[0]
    }


@torch.inference_mode()
def _validate_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_function: DNS3Loss,
    device: torch.device,
    fp16: bool,
) -> dict[str, float]:
    model.eval()
    rows, weights = [], []
    for batch in loader:
        noisy = batch["noisy"].float().to(device, non_blocking=True)
        clean = batch["clean"].float().to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=fp16):
            loss, terms, _ = loss_function(model, noisy, clean)
        rows.append(
            {
                "loss": float(loss),
                **{key: float(value) for key, value in terms.items()},
            }
        )
        weights.append(int(noisy.shape[0]))
    return _mean(rows, weights)


@torch.inference_mode()
def _dev_pesq(
    model: nn.Module,
    pairs: list[tuple[Path, Path]],
    sample_rate: int,
    device: torch.device,
    workers: int,
    fp16: bool,
) -> float:
    import soundfile as sf

    model.eval()
    scores = []

    def score(values: tuple[np.ndarray, np.ndarray]) -> float:
        return intrusive_metrics(values[0], values[1], sample_rate, enabled=("PESQ",))["PESQ"]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = []
        for noisy_path, clean_path in pairs:
            noisy, noisy_rate = sf.read(noisy_path, dtype="float32")
            clean, clean_rate = sf.read(clean_path, dtype="float32")
            if noisy_rate != sample_rate or clean_rate != sample_rate:
                raise ValueError(f"sample-rate mismatch for {noisy_path}")
            length = min(len(noisy), len(clean))
            tensor = torch.from_numpy(noisy[:length]).unsqueeze(0).to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=fp16):
                enhanced = model(tensor)[0].float().cpu().numpy()
            if enhanced.shape != clean[:length].shape:
                raise RuntimeError("dev enhancement length drift")
            pending.append((clean[:length], enhanced))
        scores.extend(pool.map(score, pending))
    return float(np.nanmean(scores))


def load_config(path: Path, overrides: list[str]) -> Any:
    config = OmegaConf.load(path)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(config)
    return config


def train(
    config: Any,
    *,
    audit_only: bool = False,
    model_entry: str | None = None,
) -> dict[str, Any]:
    seed = int(config.training.seed)
    _seed_everything(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("medium")

    configured_model = str(config.model.name)
    selected_model = model_entry or configured_model
    model = build_model(selected_model)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    audit = {
        "model": selected_model,
        "base_config_model": configured_model,
        "parameters": parameters,
    }
    if hasattr(model, "physical_recurrent_banks"):
        audit["physical_recurrent_banks"] = model.physical_recurrent_banks
    if audit_only:
        return audit
    pairs = {
        split: discover_pairs(
            config.data[split].noisy_dir,
            config.data[split].clean_dir,
        )
        for split in ("train", "dev", "test")
    }
    audit["pair_counts"] = {key: len(value) for key, value in pairs.items()}

    device = torch.device(str(config.training.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but unavailable")
    model.to(device)
    optimizer, parameter_groups = build_optimizer(
        model,
        lr=float(config.training.optimizer.lr),
        weight_decay=float(config.training.optimizer.weight_decay),
        wd_ratio=float(config.training.optimizer.wd_ratio),
        betas=tuple(float(value) for value in config.training.optimizer.betas),
    )
    scheduler = CosineAnnealingWarmup(
        optimizer,
        warmup_iterations=int(config.training.scheduler.warmup_iterations),
        epochs=int(config.training.epochs),
        eta_min=float(config.training.scheduler.eta_min),
    )
    fp16 = bool(config.training.fp16)
    scaler = torch.amp.GradScaler(device.type, enabled=fp16)
    loss_function = DNS3Loss(
        mag_mse=float(config.loss.mag_mse),
        complex_mse=float(config.loss.complex_mse),
        consistency=float(config.loss.consistency),
        wav_l1=float(config.loss.wav_l1),
        compression=float(config.frontend.compression),
    )

    segment_samples = int(float(config.data.segment_seconds) * int(config.data.sample_rate))
    train_dataset = DNS3SegmentDataset(
        pairs["train"],
        sample_rate=int(config.data.sample_rate),
        segment_samples=segment_samples,
        seed=seed,
        random_crop=True,
    )
    sampler = EpochSubsetSampler(
        len(train_dataset),
        int(config.data.train.samples_per_epoch),
        seed,
        replacement=bool(config.data.train.sampling_with_replacement),
    )
    workers = int(config.training.num_workers)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config.training.batch_size),
        sampler=sampler,
        num_workers=workers,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
        persistent_workers=bool(config.training.persistent_workers) and workers > 0,
        pin_memory=True,
        drop_last=True,
    )
    valid_dataset = DNS3SegmentDataset(
        pairs["dev"],
        sample_rate=int(config.data.sample_rate),
        segment_samples=segment_samples,
        seed=seed + 100_000,
        random_crop=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(config.validation.batch_size),
        num_workers=int(config.validation.num_workers),
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed + 100_000),
        pin_memory=True,
    )

    output_dir = Path(str(config.training.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_dir / "config.yaml")
    (output_dir / "run.json").write_text(
        json.dumps({**audit, "optimizer_groups": parameter_groups}, indent=2) + "\n",
        encoding="utf-8",
    )
    latest = output_dir / "latest.pt"
    start_epoch, history = 0, []
    best_pesq, best_epoch = -float("inf"), -1
    if bool(config.training.resume) and latest.is_file():
        checkpoint = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"])
        history = list(checkpoint["history"])
        best_pesq = float(checkpoint["best_dev_pesq"])
        best_epoch = int(checkpoint["best_dev_epoch"])
        _restore_rng(checkpoint)

    epochs = int(config.training.epochs)
    log_interval = int(config.training.log_interval)
    for epoch in range(start_epoch + 1, epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        rows = []
        started = time.perf_counter()
        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            noisy = batch["noisy"].float().to(device, non_blocking=True)
            clean = batch["clean"].float().to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=fp16):
                loss, terms, _ = loss_function(model, noisy, clean)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at E{epoch} step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            scheduler.warmup_step()
            rows.append(
                {
                    "loss": float(loss.detach()),
                    **{key: float(value.detach()) for key, value in terms.items()},
                }
            )
            if step == 1 or step % log_interval == 0 or step == len(train_loader):
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": step,
                            "steps": len(train_loader),
                            "lr": optimizer.param_groups[0]["lr"],
                            **_mean(rows[-log_interval:]),
                        }
                    ),
                    flush=True,
                )
        scheduler.step()
        valid = None
        dev_pesq = None
        if epoch % int(config.validation.interval) == 0:
            valid = _validate_loss(model, valid_loader, loss_function, device, fp16)
        if epoch % int(config.selection.interval) == 0:
            dev_pesq = _dev_pesq(
                model,
                pairs["dev"],
                int(config.data.sample_rate),
                device,
                int(config.selection.workers),
                fp16,
            )
            if dev_pesq > best_pesq:
                best_pesq, best_epoch = dev_pesq, epoch
        record = {
            "epoch": epoch,
            "seconds": time.perf_counter() - started,
            "lr_next_epoch": optimizer.param_groups[0]["lr"],
            "train": _mean(rows),
            "valid": valid,
            "dev_pesq": dev_pesq,
        }
        history.append(record)
        payload = _checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            history=history,
            best_pesq=best_pesq,
            best_epoch=best_epoch,
        )
        _atomic_save(payload, latest)
        if best_epoch == epoch:
            _atomic_save(payload, output_dir / "best-dev-pesq.pt")
        if epoch % int(config.training.checkpoint_interval) == 0:
            _atomic_save(payload, output_dir / f"epoch-{epoch:05d}.pt")
        print(json.dumps(record), flush=True)
    result = {
        "status": "completed",
        "model": selected_model,
        "base_config_model": configured_model,
        "best_dev_pesq": best_pesq,
        "best_dev_epoch": best_epoch,
        "selected_checkpoint": str(output_dir / "best-dev-pesq.pt"),
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--audit", action="store_true")
    result.add_argument(
        "--model-entry",
        help="model module name; defaults to model.name in the config",
    )
    result.add_argument("overrides", nargs="*", help="OmegaConf dot-list overrides")
    return result


def main() -> None:
    args = parser().parse_args()
    config = load_config(args.config, args.overrides)
    print(
        json.dumps(
            train(config, audit_only=args.audit, model_entry=args.model_entry),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

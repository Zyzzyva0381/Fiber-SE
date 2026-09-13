from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Optimizer


def compressed_stft(
    waveform: Tensor,
    *,
    n_fft: int = 512,
    hop_size: int = 256,
    win_size: int = 512,
    compression: float = 0.3,
    discard_nyquist: bool,
    epsilon: float = 1.0e-5,
) -> Tensor:
    window = torch.hann_window(win_size, device=waveform.device, dtype=waveform.dtype)
    spectrum = torch.stft(
        waveform,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    if discard_nyquist:
        spectrum = spectrum[:, :-1]
    magnitude = spectrum.abs().unsqueeze(-1).clamp_min(epsilon)
    return torch.view_as_real(spectrum) * magnitude.pow(compression - 1.0)


class DNS3Loss(nn.Module):
    def __init__(
        self,
        *,
        mag_mse: float = 0.3,
        complex_mse: float = 0.2,
        consistency: float = 0.3,
        wav_l1: float = 0.2,
        compression: float = 0.3,
    ) -> None:
        super().__init__()
        self.weights = {
            "mag_mse": float(mag_mse),
            "complex_mse": float(complex_mse),
            "consistency": float(consistency),
            "wav_l1": float(wav_l1),
        }
        self.compression = float(compression)

    def forward(
        self, model: nn.Module, noisy: Tensor, clean: Tensor
    ) -> tuple[Tensor, dict[str, Tensor], Tensor]:
        enhanced, enhanced_compressed = model.training_forward(noisy)
        clean = clean[..., : enhanced.shape[-1]]
        clean_compressed = compressed_stft(
            clean,
            compression=self.compression,
            discard_nyquist=True,
        )
        terms = {
            "mag_mse": F.mse_loss(
                torch.linalg.vector_norm(enhanced_compressed, dim=-1),
                torch.linalg.vector_norm(clean_compressed, dim=-1),
            ),
            "complex_mse": F.mse_loss(enhanced_compressed, clean_compressed),
            "consistency": F.mse_loss(
                compressed_stft(
                    enhanced,
                    compression=self.compression,
                    discard_nyquist=False,
                ),
                compressed_stft(
                    clean,
                    compression=self.compression,
                    discard_nyquist=False,
                ),
            ),
            "wav_l1": F.l1_loss(enhanced, clean),
        }
        loss = sum(self.weights[name] * terms[name] for name in self.weights)
        return loss, terms, enhanced


def _projection_channel(parameter: Tensor, update: Tensor, epsilon: float) -> Tensor:
    normalized = parameter / parameter.norm(p=2, dim=1, keepdim=True).add(epsilon)
    projected = (normalized * update).sum(dim=1, keepdim=True)
    return update - normalized * projected


def _projection_layer(parameter: Tensor, update: Tensor, epsilon: float) -> Tensor:
    normalized = parameter / parameter.norm(p=2, dim=0, keepdim=True).add(epsilon)
    projected = (normalized * update).sum(dim=0, keepdim=True)
    return update - normalized * projected


class AdamP(Optimizer):


    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        *,
        lr: float,
        betas: tuple[float, float],
        weight_decay: float,
        wd_ratio: float,
        projection: str = "auto",
        eps: float = 1.0e-8,
        delta: float = 0.1,
    ) -> None:
        super().__init__(
            params,
            dict(
                lr=float(lr),
                betas=tuple(float(value) for value in betas),
                weight_decay=float(weight_decay),
                wd_ratio=float(wd_ratio),
                projection=str(projection),
                eps=float(eps),
                delta=float(delta),
            ),
        )

    @staticmethod
    def _auto_projection(
        parameter: Tensor,
        update: Tensor,
        delta: float,
        wd_ratio: float,
        epsilon: float,
    ) -> tuple[Tensor, float]:
        if parameter.ndim > 1:
            value = parameter.data.reshape(parameter.size(0), -1)
            perturbation = update.reshape(parameter.size(0), -1)
            cosine = F.cosine_similarity(value, perturbation, dim=1, eps=epsilon).abs_()
            if cosine.max() < delta / math.sqrt(value.size(1)):
                return (
                    _projection_channel(value, perturbation, epsilon).reshape_as(parameter),
                    wd_ratio,
                )
        value = parameter.data.reshape(-1)
        perturbation = update.reshape(-1)
        cosine = F.cosine_similarity(value, perturbation, dim=0, eps=epsilon).abs_()
        if cosine.max() < delta / math.sqrt(value.size(0)):
            return (
                _projection_layer(value, perturbation, epsilon).reshape_as(parameter),
                wd_ratio,
            )
        return update, 1.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                average = state["exp_avg"]
                square_average = state["exp_avg_sq"]
                average.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                square_average.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                correction1 = 1.0 - beta1 ** state["step"]
                correction2 = 1.0 - beta2 ** state["step"]
                denominator = square_average.sqrt().div_(math.sqrt(correction2)).add_(group["eps"])
                update = average / denominator
                if parameter.numel() == 1 or group["projection"] == "disabled":
                    decay_ratio = 1.0
                elif group["projection"] == "auto":
                    update, decay_ratio = self._auto_projection(
                        parameter,
                        update,
                        group["delta"],
                        group["wd_ratio"],
                        group["eps"],
                    )
                elif group["projection"] == "channelwise":
                    shape = parameter.shape
                    update = _projection_channel(
                        parameter.data.reshape(parameter.size(0), -1),
                        update.reshape(parameter.size(0), -1),
                        group["eps"],
                    ).reshape(shape)
                    decay_ratio = group["wd_ratio"]
                elif group["projection"] == "layerwise":
                    update = _projection_layer(
                        parameter.data.reshape(-1),
                        update.reshape(-1),
                        group["eps"],
                    ).reshape_as(parameter)
                    decay_ratio = group["wd_ratio"]
                else:
                    raise ValueError(f"unsupported AdamP projection {group['projection']}")
                if group["weight_decay"] > 0.0:
                    parameter.mul_(1.0 - group["lr"] * group["weight_decay"] * decay_ratio)
                parameter.add_(update, alpha=-(group["lr"] / correction1))
        return loss


_GROUPS = (
    (
        "scale_no_decay",
        (
            r"blocks\.\d+\.temporal\.parametrizations\.weight_(?:ih|hh)_l0\.original0$",
            r"rf_block\.\d+\.rnn\.parametrizations\.weight_(?:ih|hh)_l0\.original0$",
            r"output_projection\.3\.scale$",
            r"dec_post\.3\.scale$",
        ),
        {"weight_decay": 0.0, "projection": "disabled"},
    ),
    (
        "channelwise",
        (
            r".+parametrizations.+original1$",
            r"input\.0\.weight$",
            r"encoder\.\d+\.0\.weight$",
            r"state_(?:in|out)\.0\.weight$",
            r"blocks\.\d+\.(?:temporal|spectral)_projection\.weight$",
            r"decoder\.\d+\.[03]\.weight$",
            r"output_projection\.0\.weight$",
            r"enc_pre\.0\.weight$",
            r"rf_(?:pre|post)\.1\.weight$",
            r"rf_block\.\d+\.(?:rnn|attn)_fc\.weight$",
            r"dec_post\.0\.weight$",
        ),
        {"projection": "channelwise"},
    ),
    (
        "layerwise",
        (
            r"output_projection\.3\.weight$",
            r"rf_(?:pre|post)\.0\.weight$",
            r"dec_post\.3\.weight$",
        ),
        {"projection": "layerwise"},
    ),
)


def build_optimizer(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    wd_ratio: float,
    betas: tuple[float, float],
) -> tuple[AdamP, dict[str, list[str]]]:
    groups: dict[str, dict[str, Any]] = {
        "default": {"params": []},
        **{name: {"params": [], **overrides} for name, _, overrides in _GROUPS},
    }
    manifest = {name: [] for name in groups}
    for parameter_name, parameter in model.named_parameters():
        matches = [
            name
            for name, patterns, _ in _GROUPS
            if any(re.search(pattern, parameter_name) for pattern in patterns)
        ]
        if len(matches) > 1:
            raise RuntimeError(f"multiple optimizer groups match {parameter_name}")
        name = matches[0] if matches else "default"
        groups[name]["params"].append(parameter)
        manifest[name].append(parameter_name)
    optimizer = AdamP(
        [group for group in groups.values() if group["params"]],
        lr=lr,
        weight_decay=weight_decay,
        wd_ratio=wd_ratio,
        betas=betas,
        projection="auto",
    )
    return optimizer, manifest


__all__ = ["AdamP", "DNS3Loss", "build_optimizer", "compressed_stft"]

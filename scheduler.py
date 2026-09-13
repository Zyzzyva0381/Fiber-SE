from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmup(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        *,
        warmup_iterations: int,
        epochs: int,
        eta_min: float,
    ) -> None:
        self.current_iteration = 1
        self.warmup_iterations = int(warmup_iterations)
        self.T_max = int(epochs)
        self.eta_min = float(eta_min)
        super().__init__(optimizer)

    def warmup_step(self) -> None:
        if self.current_iteration > self.warmup_iterations:
            return
        scale = self.current_iteration / self.warmup_iterations
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * scale
        self.current_iteration += 1

    def get_lr(self) -> list[float]:
        if self.current_iteration <= self.warmup_iterations:
            scale = self.current_iteration / self.warmup_iterations
            return [base_lr * scale for base_lr in self.base_lrs]
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1.0 + math.cos(self.last_epoch * math.pi / self.T_max))
            / 2.0
            for base_lr in self.base_lrs
        ]

    def step(self, epoch: int | None = None) -> None:
        del epoch
        if self.last_epoch == -1 or self.current_iteration > self.warmup_iterations:
            super().step()
        else:
            self.T_max -= 1


__all__ = ["CosineAnnealingWarmup"]

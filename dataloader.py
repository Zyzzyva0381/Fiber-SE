from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler


Pair = tuple[Path, Path]
EpochIndex = tuple[int, int, int]


def discover_pairs(
    noisy_dir: str | Path,
    clean_dir: str | Path,
) -> list[Pair]:
    noisy_dir = Path(noisy_dir).expanduser().resolve()
    clean_dir = Path(clean_dir).expanduser().resolve()
    noisy_paths = sorted(noisy_dir.glob("*.wav"))
    if not noisy_paths:
        raise FileNotFoundError(f"no .wav files in {noisy_dir}")
    pairs = []
    for noisy_path in noisy_paths:
        clean_path = clean_dir / noisy_path.name
        if not clean_path.is_file():
            raise FileNotFoundError(f"missing clean pair for {noisy_path}")
        pairs.append((noisy_path, clean_path))
    return pairs


def _read_audio(path: Path, sample_rate: int) -> np.ndarray:
    waveform, rate = sf.read(path, dtype="float32", always_2d=False)
    if int(rate) != sample_rate:
        raise ValueError(f"expected {sample_rate} Hz but {path} is {rate} Hz")
    if waveform.ndim != 1:
        waveform = waveform.mean(axis=-1, dtype=np.float32)
    return np.asarray(waveform, dtype=np.float32)


class DNS3SegmentDataset(Dataset):


    def __init__(
        self,
        pairs: list[Pair],
        *,
        sample_rate: int,
        segment_samples: int,
        seed: int,
        random_crop: bool,
    ) -> None:
        self.pairs = list(pairs)
        self.sample_rate = int(sample_rate)
        self.segment_samples = int(segment_samples)
        self.seed = int(seed)
        self.random_crop = bool(random_crop)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, key: int | EpochIndex) -> dict[str, Tensor | str]:
        if isinstance(key, tuple):
            epoch, index, draw_index = (int(value) for value in key)
        else:
            epoch, index, draw_index = 0, int(key), int(key)
        noisy_path, clean_path = self.pairs[index]
        noisy = _read_audio(noisy_path, self.sample_rate)
        clean = _read_audio(clean_path, self.sample_rate)
        available = min(len(noisy), len(clean))
        noisy, clean = noisy[:available], clean[:available]
        if available < self.segment_samples:
            padding = self.segment_samples - available
            left = padding // 2
            noisy = np.pad(noisy, (left, padding - left))
            clean = np.pad(clean, (left, padding - left))
        elif available > self.segment_samples:
            if self.random_crop:
                generator = random.Random(
                    self.seed + epoch * 1_000_003 + index * 10_007 + draw_index
                )
                start = generator.randrange(available - self.segment_samples + 1)
            else:
                start = (available - self.segment_samples) // 2
            noisy = noisy[start : start + self.segment_samples]
            clean = clean[start : start + self.segment_samples]
        return {
            "uid": noisy_path.stem,
            "noisy": torch.from_numpy(noisy.copy()),
            "clean": torch.from_numpy(clean.copy()),
        }


class EpochSubsetSampler(Sampler[EpochIndex]):


    def __init__(
        self,
        dataset_size: int,
        samples_per_epoch: int,
        seed: int,
        *,
        replacement: bool = False,
    ) -> None:
        if dataset_size <= 0 or samples_per_epoch <= 0:
            raise ValueError("dataset_size and samples_per_epoch must be positive")
        if samples_per_epoch > dataset_size and not replacement:
            raise ValueError("cannot sample beyond the pool without replacement")
        self.dataset_size = int(dataset_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.replacement = bool(replacement)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def indices_for_epoch(self, epoch: int) -> list[int]:
        generator = torch.Generator().manual_seed(self.seed + int(epoch))
        if self.replacement:
            selected = torch.randint(
                self.dataset_size,
                (self.samples_per_epoch,),
                generator=generator,
            )
        else:
            selected = torch.randperm(self.dataset_size, generator=generator)[
                : self.samples_per_epoch
            ]
        return [int(index) for index in selected]

    def __iter__(self) -> Iterator[EpochIndex]:
        return iter(
            (self.epoch, index, draw_index)
            for draw_index, index in enumerate(self.indices_for_epoch(self.epoch))
        )

    def __len__(self) -> int:
        return self.samples_per_epoch


def seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


__all__ = [
    "DNS3SegmentDataset",
    "EpochSubsetSampler",
    "discover_pairs",
    "seed_worker",
]

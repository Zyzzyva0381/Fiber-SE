from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import librosa
import numpy as np


def _aligned(reference: np.ndarray, enhanced: np.ndarray):
    length = min(len(reference), len(enhanced))
    return (
        np.asarray(reference[:length], dtype=np.float32),
        np.asarray(enhanced[:length], dtype=np.float32),
    )


def sdr(reference: np.ndarray, enhanced: np.ndarray) -> float:
    reference, enhanced = _aligned(reference, enhanced)
    reference = reference - reference.mean()
    enhanced = enhanced - enhanced.mean()
    return float(
        10.0
        * np.log10((np.sum(reference**2) + 1.0e-8) / (np.sum((enhanced - reference) ** 2) + 1.0e-8))
    )


def sisnr(reference: np.ndarray, enhanced: np.ndarray) -> float:
    reference, enhanced = _aligned(reference, enhanced)
    reference = reference - reference.mean()
    enhanced = enhanced - enhanced.mean()
    scale = np.sum(enhanced * reference) / (np.sum(reference**2) + 1.0e-8)
    target = scale * reference
    residual = enhanced - target
    return float(10.0 * np.log10((np.sum(target**2) + 1.0e-8) / (np.sum(residual**2) + 1.0e-8)))


def intrusive_metrics(
    reference: np.ndarray,
    enhanced: np.ndarray,
    sample_rate: int,
    enabled: Sequence[str] = ("SDR", "SISNR", "PESQ", "ESTOI", "STOI"),
) -> dict[str, float]:
    reference, enhanced = _aligned(reference, enhanced)
    requested = {name.upper() for name in enabled}
    scores: dict[str, float] = {}
    if "SDR" in requested:
        scores["SDR"] = sdr(reference, enhanced)
    if "SISNR" in requested:
        scores["SISNR"] = sisnr(reference, enhanced)
    if "PESQ" in requested:
        from pesq import PesqError, pesq

        rate = int(sample_rate)
        ref, enh = reference, enhanced
        if rate > 16000:
            ref = librosa.resample(ref, orig_sr=rate, target_sr=16000)
            enh = librosa.resample(enh, orig_sr=rate, target_sr=16000)
            rate = 16000
        if rate not in (8000, 16000):
            raise ValueError("PESQ requires 8 kHz or 16 kHz")
        value = pesq(
            rate,
            ref,
            enh,
            "nb" if rate == 8000 else "wb",
            on_error=PesqError.RETURN_VALUES,
        )
        scores["PESQ"] = float(value) if value != PesqError.NO_UTTERANCES_DETECTED else float("nan")
    if requested & {"ESTOI", "STOI"}:
        from pystoi.stoi import stoi

        if "ESTOI" in requested:
            scores["ESTOI"] = float(stoi(reference, enhanced, fs_sig=sample_rate, extended=True))
        if "STOI" in requested:
            scores["STOI"] = float(stoi(reference, enhanced, fs_sig=sample_rate, extended=False))
    return scores


class DNSMOSMetrics:


    INPUT_LENGTH = 9.01
    SAMPLE_RATE = 16000

    def __init__(self, primary_model: Path, p808_model: Path) -> None:
        import onnxruntime as ort

        self.primary = ort.InferenceSession(str(primary_model), providers=["CPUExecutionProvider"])
        self.p808 = ort.InferenceSession(str(p808_model), providers=["CPUExecutionProvider"])

    @staticmethod
    def _polyfit(sig: float, bak: float, ovr: float) -> tuple[float, float, float]:
        return (
            float(np.poly1d([-0.08397278, 1.22083953, 0.0052439])(sig)),
            float(np.poly1d([-0.13166888, 1.60915514, -0.39604546])(bak)),
            float(np.poly1d([-0.06766283, 1.11546468, 0.04602535])(ovr)),
        )

    @staticmethod
    def _melspec(audio: np.ndarray) -> np.ndarray:
        values = librosa.feature.melspectrogram(
            y=audio,
            sr=16000,
            n_fft=321,
            hop_length=160,
            n_mels=120,
        )
        values = (librosa.power_to_db(values, ref=np.max) + 40) / 40
        return values.T

    def __call__(self, waveform: np.ndarray, sample_rate: int) -> dict[str, float]:
        audio = np.asarray(waveform, dtype=np.float32)
        if int(sample_rate) != self.SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.SAMPLE_RATE)
        length = int(self.INPUT_LENGTH * self.SAMPLE_RATE)
        while len(audio) < length:
            audio = np.append(audio, audio)
        hops = int(np.floor(len(audio) / self.SAMPLE_RATE) - self.INPUT_LENGTH) + 1
        values = {key: [] for key in ("OVRL", "SIG", "BAK", "P808_MOS")}
        for index in range(hops):
            segment = audio[
                index * self.SAMPLE_RATE : int((index + self.INPUT_LENGTH) * self.SAMPLE_RATE)
            ]
            if len(segment) < length:
                continue
            primary_input = segment.astype("float32")[None]
            p808_input = self._melspec(segment[:-160]).astype("float32")[None]
            p808 = self.p808.run(None, {"input_1": p808_input})[0][0][0]
            sig_raw, bak_raw, ovr_raw = self.primary.run(None, {"input_1": primary_input})[0][0]
            sig, bak, ovr = self._polyfit(sig_raw, bak_raw, ovr_raw)
            values["SIG"].append(sig)
            values["BAK"].append(bak)
            values["OVRL"].append(ovr)
            values["P808_MOS"].append(float(p808))
        return {key: float(np.asarray(rows).mean()) for key, rows in values.items()}


__all__ = ["DNSMOSMetrics", "intrusive_metrics", "sdr", "sisnr"]

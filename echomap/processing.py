from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import signal


@dataclass
class EchoResponse:
    correlation: np.ndarray
    delays: np.ndarray
    aligned_delay_s: float
    direct_index: int
    unreliable: bool


def matched_filter(recording: np.ndarray, reference: np.ndarray, sample_rate: int, direct_search_seconds: float = 0.08) -> EchoResponse:
    recording = np.asarray(recording, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    correlation = signal.fftconvolve(recording, reference[::-1], mode="full")
    lags = np.arange(-len(reference) + 1, len(recording))
    positive = lags >= 0
    positive_corr = correlation[positive]
    positive_lags = lags[positive]
    search_end = min(len(positive_corr), max(1, int(direct_search_seconds * sample_rate)))
    direct_index = int(np.argmax(np.abs(positive_corr[:search_end])))
    peak = float(np.max(np.abs(positive_corr), initial=0.0))
    noise = float(np.median(np.abs(positive_corr)) + 1e-12)
    unreliable = peak < 8 * noise or not np.isfinite(peak)
    return EchoResponse(positive_corr, positive_lags / sample_rate, float(positive_lags[direct_index] / sample_rate), direct_index, unreliable)


def align_to_direct(response: EchoResponse, target_delay_s: float = 0.0) -> np.ndarray:
    return response.correlation[np.argsort(np.abs(response.delays - target_delay_s))]


def baseline_difference(current: np.ndarray, baseline: np.ndarray | None) -> np.ndarray:
    current = np.asarray(current, dtype=np.float64)
    if baseline is None or len(baseline) == 0:
        return current.copy()
    length = min(len(current), len(baseline))
    return current[:length] - baseline[:length]


def suppress_direct_path(correlation: np.ndarray, direct_index: int, guard_samples: int) -> np.ndarray:
    result = np.asarray(correlation, dtype=np.float64).copy()
    start = max(0, direct_index - guard_samples)
    end = min(len(result), direct_index + guard_samples + 1)
    result[start:end] = 0.0
    return result


def response_to_heatmap(response: EchoResponse, width: int = 240) -> np.ndarray:
    values = np.abs(response.correlation)
    if len(values) < width:
        return np.pad(values, (0, width - len(values)))
    indices = np.linspace(0, len(values) - 1, width).astype(int)
    return values[indices]

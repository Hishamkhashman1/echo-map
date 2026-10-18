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
    raw_direct_index: int
    direct_peak: float
    correlation_noise: float
    direct_snr: float
    late_peak: float
    late_snr: float
    reflection_ratio: float
    reflection_valid: bool
    reflection_reason: str


def matched_filter(recording: np.ndarray, reference: np.ndarray, sample_rate: int, direct_search_seconds: float = 0.35) -> EchoResponse:
    recording = np.asarray(recording, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    correlation = signal.fftconvolve(recording, reference[::-1], mode="full")
    lags = np.arange(-len(reference) + 1, len(recording))
    positive = lags >= 0
    positive_corr = correlation[positive]
    positive_lags = lags[positive]
    search_end = min(len(positive_corr), max(1, int(direct_search_seconds * sample_rate)))
    direct_index = int(np.argmax(np.abs(positive_corr[:search_end])))
    direct_peak = float(np.abs(positive_corr[direct_index]))
    aligned_correlation = positive_corr[direct_index:].copy()
    guard_samples = max(1, int(0.004 * sample_rate))
    late = np.abs(aligned_correlation[guard_samples:])
    late_center = float(np.median(late)) if len(late) else 0.0
    mad = float(1.4826 * np.median(np.abs(late - late_center))) if len(late) else 0.0
    noise = max(mad, float(np.percentile(late, 25)) if len(late) else 0.0, 1e-12)
    direct_snr = direct_peak / noise
    late_peak = float(np.max(late, initial=0.0))
    late_snr = late_peak / noise
    reference_autocorrelation = np.abs(signal.fftconvolve(reference, reference[::-1], mode="full"))
    reference_peak = float(np.max(reference_autocorrelation, initial=0.0))
    reference_sidelobe = float(np.max(reference_autocorrelation[len(reference) + guard_samples:], initial=0.0)) / (reference_peak + 1e-12)
    reflection_ratio = late_peak / (direct_peak + 1e-12)
    unreliable = direct_snr < 10 or direct_peak <= 1e-10 or not np.isfinite(direct_peak)
    reflection_valid = not unreliable and late_snr >= 8 and reflection_ratio >= max(0.02, 1.25 * reference_sidelobe)
    if unreliable:
        reflection_reason = "No reliable transmitted chirp/direct path was detected."
    elif not reflection_valid:
        reflection_reason = "The transmitted chirp is present, but no late reflection rises above the matched-filter sidelobe/noise gate."
    else:
        reflection_reason = "Late reflection evidence passed the direct-path and noise gates."
    aligned_delays = np.arange(len(aligned_correlation), dtype=float) / sample_rate
    return EchoResponse(aligned_correlation, aligned_delays, float(positive_lags[direct_index] / sample_rate), 0, unreliable, direct_index, direct_peak, noise, direct_snr, late_peak, late_snr, reflection_ratio, reflection_valid, reflection_reason)


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

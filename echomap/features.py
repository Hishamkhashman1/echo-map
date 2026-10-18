from __future__ import annotations

import numpy as np
from scipy import signal

FEATURE_SCHEMA = ["energy", "peak", "peak_delay_s", "late_energy", "change_energy", "spectral_centroid_hz", "temporal_variation"]


def extract_features(response, difference: np.ndarray | None, sample_rate: int, direct_guard_s: float = 0.004) -> np.ndarray:
    corr = np.asarray(response.correlation, dtype=np.float64)
    magnitude = np.abs(corr)
    guard = int(direct_guard_s * sample_rate)
    usable = magnitude.copy()
    usable[: response.direct_index + guard] = 0.0
    peak_index = int(np.argmax(usable)) if len(usable) else 0
    late = usable[response.direct_index + guard :]
    frequencies, power = signal.periodogram(corr, fs=sample_rate)
    centroid = float(np.sum(frequencies * power) / (np.sum(power) + 1e-12))
    change_energy = float(np.sqrt(np.mean(np.asarray(difference, dtype=np.float64) ** 2))) if difference is not None and len(difference) else 0.0
    temporal = float(np.std(np.diff(magnitude))) if len(magnitude) > 1 else 0.0
    values = [
        float(np.sqrt(np.mean(usable**2))) if len(usable) else 0.0,
        float(usable[peak_index]) if len(usable) else 0.0,
        float(response.delays[peak_index]) if len(usable) else 0.0,
        float(np.sqrt(np.mean(late**2))) if len(late) else 0.0,
        change_energy,
        centroid,
        temporal,
    ]
    return np.asarray(values, dtype=np.float64)

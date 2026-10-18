from __future__ import annotations

import numpy as np


def synthetic_recording(reference: np.ndarray, sample_rate: int, delays_s=(0.012, 0.041), gains=(1.0, 0.28), noise=0.002, movement=0.0, seed=2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    length = len(reference) + int(max(delays_s) * sample_rate) + 1
    result = np.zeros(length, dtype=np.float64)
    for delay, gain in zip(delays_s, gains):
        index = int(round(delay * sample_rate))
        result[index : index + len(reference)] += gain * reference
    if movement:
        index = int(round((delays_s[-1] + movement) * sample_rate))
        result[index : index + len(reference)] += gains[-1] * 0.7 * reference
    result += rng.normal(0.0, noise, length)
    return result.astype(np.float32)

from __future__ import annotations

import numpy as np


def linear_chirp(sample_rate: int, duration: float, start_hz: float, end_hz: float) -> np.ndarray:
    samples = max(1, int(round(sample_rate * duration)))
    time = np.arange(samples, dtype=np.float64) / sample_rate
    phase = 2 * np.pi * (start_hz * time + (end_hz - start_hz) * time**2 / (2 * duration))
    window = np.hanning(samples)
    return (np.sin(phase) * window).astype(np.float32)


def make_burst(sample_rate: int, duration: float, start_hz: float, end_hz: float, level: float) -> np.ndarray:
    return np.clip(linear_chirp(sample_rate, duration, start_hz, end_hz) * level, -0.99, 0.99)

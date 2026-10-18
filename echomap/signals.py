from __future__ import annotations

import numpy as np


PROBE_PHASE_CODE = np.asarray([1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1], dtype=np.float64)


def linear_chirp(sample_rate: int, duration: float, start_hz: float, end_hz: float) -> np.ndarray:
    samples = max(1, int(round(sample_rate * duration)))
    time = np.arange(samples, dtype=np.float64) / sample_rate
    phase = 2 * np.pi * (start_hz * time + (end_hz - start_hz) * time**2 / (2 * duration))
    window = np.hanning(samples)
    return (np.sin(phase) * window).astype(np.float32)


def make_burst(sample_rate: int, duration: float, start_hz: float, end_hz: float, level: float) -> np.ndarray:
    chirp = linear_chirp(sample_rate, duration, start_hz, end_hz).astype(np.float64)
    chip_indices = np.minimum((np.arange(len(chirp)) * len(PROBE_PHASE_CODE) // len(chirp)), len(PROBE_PHASE_CODE) - 1)
    coded_chirp = chirp * PROBE_PHASE_CODE[chip_indices]
    return np.clip(coded_chirp * level, -0.99, 0.99).astype(np.float32)

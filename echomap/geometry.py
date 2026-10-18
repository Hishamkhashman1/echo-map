from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class GeometryEstimate:
    candidate_paths_m: np.ndarray
    room_width_m: tuple[float, float] | None
    room_length_m: tuple[float, float] | None
    supported: bool
    explanation: str


def path_lengths_from_delays(delays_s: np.ndarray, speed_m_s: float = 343.0, timing_uncertainty_s: float = 0.001) -> tuple[np.ndarray, np.ndarray]:
    delays_s = np.asarray(delays_s, dtype=float)
    return delays_s * speed_m_s, np.full_like(delays_s, timing_uncertainty_s * speed_m_s)


def estimate_rectangular_room(delays_s: np.ndarray, speed_m_s: float = 343.0, timing_uncertainty_s: float = 0.001) -> GeometryEstimate:
    paths, uncertainty = path_lengths_from_delays(delays_s, speed_m_s, timing_uncertainty_s)
    positive = paths[paths > 0.25]
    if len(positive) < 2:
        return GeometryEstimate(paths, None, None, False, "Insufficient separated reflections for even a tentative room estimate.")
    candidates = np.sort(positive)[:2] / 2.0
    widths = (max(0.5, candidates[0] - uncertainty[0]), candidates[0] + uncertainty[0])
    lengths = (max(widths[0], candidates[1] - uncertainty[1]), candidates[1] + uncertainty[1])
    return GeometryEstimate(paths, widths, lengths, True, "Tentative wall distances assume specular first-order reflections, known sound speed, and an unknown device position; alternate layouts remain possible.")

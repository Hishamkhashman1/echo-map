from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import numpy as np
from scipy import signal


@dataclass(frozen=True)
class RoomCandidate:
    width_m: float
    length_m: float
    height_m: float
    speaker_xy: tuple[float, float]
    microphone_xy: tuple[float, float]
    absorption: float
    score: float = float("inf")
    backend: str = "analytic"


@dataclass
class RoomInference:
    candidates: list[RoomCandidate]
    supported: bool
    explanation: str
    backend: str

    @property
    def best(self) -> RoomCandidate | None:
        return self.candidates[0] if self.candidates else None


def _analytic_rir(candidate: RoomCandidate, sample_rate: int, duration_s: float, speed_m_s: float = 343.0) -> np.ndarray:
    rir = np.zeros(max(1, int(duration_s * sample_rate)), dtype=float)
    source_x, source_y = candidate.speaker_xy
    microphone_x, microphone_y = candidate.microphone_xy
    images = [(source_x, source_y, 1.0), (-source_x, source_y, candidate.absorption), (2 * candidate.width_m - source_x, source_y, candidate.absorption), (source_x, -source_y, candidate.absorption), (source_x, 2 * candidate.length_m - source_y, candidate.absorption)]
    for image_x, image_y, gain in images:
        distance = float(np.hypot(image_x - microphone_x, image_y - microphone_y))
        index = int(round(distance / speed_m_s * sample_rate))
        if 0 <= index < len(rir) and distance > 0:
            rir[index] += gain / distance
    return rir


def _pyroomacoustics_rir(candidate: RoomCandidate, sample_rate: int, duration_s: float) -> np.ndarray:
    import pyroomacoustics as pra

    room = pra.ShoeBox([candidate.width_m, candidate.length_m, candidate.height_m], fs=sample_rate, max_order=2, absorption=candidate.absorption, air_absorption=True)
    room.add_source([candidate.speaker_xy[0], candidate.speaker_xy[1], 1.2])
    room.add_microphone([candidate.microphone_xy[0], candidate.microphone_xy[1], 1.2])
    room.compute_rir()
    rir = np.asarray(room.rir[0][0], dtype=float)
    size = max(1, int(duration_s * sample_rate))
    return np.pad(rir[:size], (0, max(0, size - len(rir))))


def simulate_room_response(reference: np.ndarray, sample_rate: int, candidate: RoomCandidate, duration_s: float, backend: str = "auto") -> tuple[np.ndarray, str]:
    used = "analytic"
    if backend in ("auto", "pyroomacoustics"):
        try:
            rir = _pyroomacoustics_rir(candidate, sample_rate, duration_s)
            used = "pyroomacoustics"
        except (ImportError, ModuleNotFoundError):
            if backend == "pyroomacoustics":
                raise
            rir = _analytic_rir(candidate, sample_rate, duration_s)
    else:
        rir = _analytic_rir(candidate, sample_rate, duration_s)
    convolved = signal.fftconvolve(rir, np.asarray(reference, dtype=float), mode="full")
    return convolved[: max(1, int(duration_s * sample_rate))].astype(np.float32), used


def _signature(response: np.ndarray, sample_rate: int, bins: int = 256) -> np.ndarray:
    magnitude = np.abs(np.asarray(response, dtype=float))
    magnitude[: int(0.004 * sample_rate)] = 0.0
    magnitude = np.log1p(magnitude / (np.max(magnitude) + 1e-12) * 100.0)
    if len(magnitude) < bins:
        magnitude = np.pad(magnitude, (0, bins - len(magnitude)))
    else:
        magnitude = magnitude[np.linspace(0, len(magnitude) - 1, bins).astype(int)]
    magnitude -= magnitude.mean()
    norm = np.linalg.norm(magnitude)
    return magnitude / norm if norm > 1e-12 else magnitude


def reflection_evidence(response: np.ndarray, sample_rate: int) -> tuple[bool, str]:
    values = np.abs(np.asarray(response, dtype=float))
    guard = max(1, int(0.004 * sample_rate))
    if len(values) <= guard or not np.all(np.isfinite(values)):
        return False, "The response is empty or non-finite."
    direct = float(np.max(values[:guard], initial=0.0))
    late = values[guard:]
    noise = max(float(1.4826 * np.median(np.abs(late - np.median(late)))), float(np.percentile(late, 25)), 1e-12)
    late_peak = float(np.max(late, initial=0.0))
    if direct <= 1e-10:
        return False, "No direct transmitted signal is present."
    if late_peak / noise < 8 or late_peak / direct < 0.02:
        return False, "No late reflection rises above the direct-path sidelobe/noise gate."
    return True, "Late reflection evidence passed the room-fitting gate."


def candidate_grid(device_separation_m: float = 0.12) -> list[RoomCandidate]:
    candidates = []
    for width_m, length_m, absorption, x_fraction, y_fraction in product((3.5, 5.0, 6.5), (4.5, 6.0, 7.5), (0.2, 0.6), (0.35, 0.5, 0.65), (0.35, 0.5, 0.65)):
        speaker_x = width_m * x_fraction
        speaker_y = length_m * y_fraction
        microphone_x = min(width_m - 0.1, speaker_x + device_separation_m)
        candidates.append(RoomCandidate(width_m, length_m, 2.5, (speaker_x, speaker_y), (microphone_x, speaker_y), absorption))
    return candidates


def fit_room_candidates(measured_response: np.ndarray, sample_rate: int, reference: np.ndarray, device_separation_m: float = 0.12, max_candidates: int = 5, backend: str = "auto", reflection_valid: bool | None = None) -> RoomInference:
    if reflection_valid is False:
        return RoomInference([], False, "Room fitting rejected: the measurement has no usable late reflection evidence.", backend)
    evidence_ok, evidence_reason = reflection_evidence(measured_response, sample_rate)
    if not evidence_ok:
        return RoomInference([], False, f"Room fitting rejected: {evidence_reason}", backend)
    measured_signature = _signature(measured_response, sample_rate)
    if np.linalg.norm(measured_signature) < 1e-12:
        return RoomInference([], False, "The aligned response has insufficient energy for model fitting.", backend)
    ranked = []
    backend_used = "analytic"
    for candidate in candidate_grid(device_separation_m):
        simulated, used = simulate_room_response(reference, sample_rate, candidate, len(measured_response) / sample_rate, backend)
        backend_used = used
        score = float(np.mean((_signature(simulated, sample_rate) - measured_signature) ** 2))
        ranked.append(RoomCandidate(**{**candidate.__dict__, "score": score, "backend": used}))
    ranked.sort(key=lambda item: item.score)
    supported = bool(ranked) and ranked[0].score < 0.9
    explanation = "Ranked rectangular-room simulation fits; orientation is arbitrary and fit score is not calibrated confidence." if supported else "No candidate room explains the measured response reliably; retain the echo evidence without a room outline."
    return RoomInference(ranked[:max_candidates], supported, explanation, backend_used)

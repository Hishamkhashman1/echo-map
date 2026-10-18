from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import signal


@dataclass
class ReflectionCandidate:
    relative_delay_s: float
    delay_uncertainty_s: float
    path_length_m: float
    path_uncertainty_m: float
    peak_amplitude: float
    repeat_count: int = 1


@dataclass
class ShellConstraint:
    relative_delay_s: float
    delay_uncertainty_s: float
    path_length_m: float
    path_uncertainty_m: float
    repeat_count: int
    colocated_approximation: bool


def extract_reflection_candidates(response, sample_rate: int, bandwidth_hz: float, sound_speed: float = 340.0, microphone_offset=(0.0, 0.0, 0.0), timing_uncertainty_s: float = 0.0005, max_candidates: int = 6, direct_ringing_guard_s: float = 0.023, max_path_length_m: float | None = None) -> list[ReflectionCandidate]:
    if response.unreliable or not response.reflection_valid:
        return []
    magnitude = np.abs(np.asarray(response.correlation, dtype=float))
    guard = max(1, int(direct_ringing_guard_s * sample_rate))
    if len(magnitude) <= guard:
        return []
    direct_distance = float(np.linalg.norm(np.asarray(microphone_offset, dtype=float)))
    late = magnitude[guard:]
    noise = max(response.correlation_noise, 1e-12)
    prominence = max(8.0 * noise, 0.03 * response.direct_peak)
    peak_indices, properties = signal.find_peaks(late, prominence=prominence, distance=max(1, int(0.002 * sample_rate)))
    uncertainty = max(1.0 / max(float(bandwidth_hz), 1.0), 2.0 / sample_rate, timing_uncertainty_s)
    ranked_peaks = sorted(peak_indices, key=lambda peak: float(properties["prominences"][np.where(peak_indices == peak)[0][0]]), reverse=True)[:max_candidates]
    candidates = []
    for peak in sorted(ranked_peaks):
        peak_amplitude = float(late[peak])
        if peak_amplitude / (response.direct_peak + 1e-12) < 0.05:
            continue
        relative_delay = float((guard + peak) / sample_rate)
        path_length = direct_distance + sound_speed * relative_delay
        if max_path_length_m is not None and path_length > max_path_length_m:
            continue
        candidates.append(ReflectionCandidate(relative_delay, uncertainty, path_length, sound_speed * uncertainty, peak_amplitude))
    return candidates


class ReflectionTracker:
    def __init__(self, sound_speed: float = 340.0, microphone_offset=(0.0, 0.0, 0.0), colocated_approximation: bool = True, minimum_repeats: int = 2, max_missed: int = 3):
        self.sound_speed = sound_speed
        self.microphone_offset = tuple(float(value) for value in microphone_offset)
        self.colocated_approximation = colocated_approximation
        self.minimum_repeats = minimum_repeats
        self.max_missed = max_missed
        self._tracks: list[list[ReflectionCandidate]] = []
        self._misses: list[int] = []

    def reset(self) -> None:
        self._tracks.clear()
        self._misses.clear()

    def update(self, observations: list[ReflectionCandidate]) -> None:
        if not observations:
            return
        matched = set()
        for observation in sorted(observations, key=lambda item: item.peak_amplitude, reverse=True):
            best_index = None
            best_delta = float("inf")
            for index, track in enumerate(self._tracks):
                if index in matched:
                    continue
                delta = abs(track[-1].relative_delay_s - observation.relative_delay_s)
                tolerance = max(track[-1].delay_uncertainty_s, observation.delay_uncertainty_s)
                if delta <= tolerance and delta < best_delta:
                    best_index, best_delta = index, delta
            if best_index is None:
                self._tracks.append([observation])
                self._misses.append(0)
                matched.add(len(self._tracks) - 1)
            else:
                self._tracks[best_index].append(observation)
                matched.add(best_index)
                self._misses[best_index] = 0
        self._misses = [0 if index in matched else misses + 1 for index, misses in enumerate(self._misses)]

    def shells(self) -> list[ShellConstraint]:
        result = []
        for track in self._tracks:
            if len(track) < self.minimum_repeats:
                continue
            delays = np.asarray([item.relative_delay_s for item in track])
            paths = np.asarray([item.path_length_m for item in track])
            delay_uncertainty = max(float(np.std(delays)), max(item.delay_uncertainty_s for item in track))
            result.append(ShellConstraint(float(np.mean(delays)), delay_uncertainty, float(np.mean(paths)), self.sound_speed * delay_uncertainty, len(track), self.colocated_approximation))
        return sorted(result, key=lambda item: item.relative_delay_s)


def shell_surface(shell: ShellConstraint, microphone_offset=(0.0, 0.0, 0.0), view_volume=((-5.0, 5.0), (-5.0, 5.0), (-3.0, 3.0)), resolution: int = 28):
    speaker = np.zeros(3, dtype=float)
    microphone = np.asarray(microphone_offset, dtype=float)
    path_length = shell.path_length_m
    focal_distance = float(np.linalg.norm(microphone - speaker))
    if shell.colocated_approximation:
        center = speaker
        major = path_length / 2.0
        minor = major
    else:
        if path_length <= focal_distance:
            return None
        center = (speaker + microphone) / 2.0
        major = path_length / 2.0
        minor = float(np.sqrt(max(0.0, major**2 - (focal_distance / 2.0) ** 2)))
    azimuth = np.linspace(0, 2 * np.pi, resolution)
    elevation = np.linspace(-np.pi / 2, np.pi / 2, resolution)
    azimuth_grid, elevation_grid = np.meshgrid(azimuth, elevation)
    local = np.stack([major * np.cos(elevation_grid) * np.cos(azimuth_grid), minor * np.cos(elevation_grid) * np.sin(azimuth_grid), minor * np.sin(elevation_grid)], axis=-1)
    if not shell.colocated_approximation and focal_distance > 1e-9:
        axis = (microphone - speaker) / focal_distance
        helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        side = np.cross(axis, helper)
        side /= np.linalg.norm(side)
        up = np.cross(axis, side)
        rotation = np.column_stack([axis, side, up])
        local = local @ rotation.T
    points = local + center
    for axis, limits in enumerate(view_volume):
        points[..., axis] = np.where((points[..., axis] >= limits[0]) & (points[..., axis] <= limits[1]), points[..., axis], np.nan)
    return points[..., 0], points[..., 1], points[..., 2]

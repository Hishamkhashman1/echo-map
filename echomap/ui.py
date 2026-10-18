from __future__ import annotations

from collections import deque
import matplotlib.pyplot as plt
import numpy as np

from .features import SEQUENCE_FEATURE_SCHEMA, aggregate_sequence_features, extract_features
from .constraints import ReflectionTracker, extract_reflection_candidates, shell_surface
from .processing import baseline_difference, matched_filter


class LiveView:
    def __init__(self, processor, reference, sample_rate: int, model=None, history_size=160, figure=None, calibration_measurements=8, bandwidth_hz: float = 7200.0, sound_speed: float = 340.0, microphone_offset=(0.0, 0.0, 0.0), microphone_offset_known: bool = False, view_volume=((-5.0, 5.0), (-5.0, 5.0), (-3.0, 3.0))):
        self.processor = processor
        self.reference = reference
        self.sample_rate = sample_rate
        self.model = model
        self.baseline = None
        self.calibration_measurements = calibration_measurements
        self.calibration_samples = []
        self.feature_history = deque(maxlen=24)
        self.last_response = None
        self.status = "calibrating"
        self.presence_status = "presence model not trained"
        self.bandwidth_hz = bandwidth_hz
        self.sound_speed = sound_speed
        self.microphone_offset = tuple(microphone_offset)
        self.microphone_offset_known = microphone_offset_known
        self.view_volume = view_volume
        self.tracker = ReflectionTracker(sound_speed, microphone_offset, not microphone_offset_known, minimum_repeats=4)
        corners = np.asarray([[x, y, z] for x in view_volume[0] for y in view_volume[1] for z in view_volume[2]], dtype=float)
        speaker = np.zeros(3, dtype=float)
        microphone = np.asarray(microphone_offset, dtype=float)
        self.max_path_length_m = float(np.max(np.linalg.norm(corners - speaker, axis=1) + np.linalg.norm(corners - microphone, axis=1)))
        self.direct_ringing_guard_s = max(0.004, min(0.012, 0.25 * len(reference) / sample_rate))
        self.current_candidates = []
        self.current_shells = []
        self.shells_stale = False
        self.measurement_index = 0
        if figure is None:
            self.fig = plt.figure(figsize=(12, 5))
            self.echo_axis = self.fig.add_subplot(1, 2, 1)
            self.room_axis = self.fig.add_subplot(1, 2, 2, projection="3d")
        else:
            self.fig = figure
            self.echo_axis = figure.add_subplot(1, 2, 1)
            self.room_axis = figure.add_subplot(1, 2, 2, projection="3d")
        self.fig.canvas.mpl_connect("key_press_event", self._key)
        self.fig.canvas.mpl_connect("close_event", lambda event: self.processor.stop())

    def _key(self, event):
        if event.key == "q":
            self.processor.stop()
            plt.close(self.fig)

    def close(self):
        self.tracker.reset()
        self.current_candidates = []
        self.current_shells = []
        self.shells_stale = False

    def update(self, recording: np.ndarray):
        self.measurement_index += 1
        response = matched_filter(recording, self.reference, self.sample_rate)
        if self.baseline is None and not response.unreliable:
            self.calibration_samples.append(response.correlation.copy())
            self.status = f"calibrating ({len(self.calibration_samples)}/{self.calibration_measurements})"
            if len(self.calibration_samples) >= self.calibration_measurements:
                length = min(len(sample) for sample in self.calibration_samples)
                self.baseline = np.mean([sample[:length] for sample in self.calibration_samples], axis=0)
                self.status = "no significant change"
        difference = baseline_difference(response.correlation, self.baseline)
        feature_vector = extract_features(response, difference, self.sample_rate)
        self.feature_history.append(feature_vector)
        if response.unreliable:
            self.status = "uncertain"
            self.presence_status = "uncertain"
            self._retain_or_expire_shells()
        elif not response.reflection_valid:
            self.status = "signal detected; no usable reflections"
            self.presence_status = "presence model not trained" if self.model is None else "uncertain"
            self._retain_or_expire_shells()
        elif self.baseline is None:
            pass
        elif self.model is not None:
            expected_schema = self.model.settings.get("feature_schema", [])
            if len(expected_schema) == len(SEQUENCE_FEATURE_SCHEMA):
                if len(self.feature_history) < 4:
                    self.presence_status = "uncertain"
                    label, reliable = "uncertain", False
                else:
                    label, _, reliable = self.model.predict(aggregate_sequence_features(np.asarray(self.feature_history)))
            else:
                label, _, reliable = self.model.predict(feature_vector)
            self.presence_status = label if reliable else "uncertain"
            self.status = "person detected" if reliable and "person" in label.lower() else (label if reliable else "uncertain")
        else:
            change = float(np.sqrt(np.mean(difference**2))) if len(difference) else 0.0
            guard = max(1, int(0.004 * self.sample_rate))
            baseline_late = self.baseline[guard:] if self.baseline is not None else np.array([])
            change_late = np.asarray(difference)[guard:]
            baseline_energy = float(np.sqrt(np.mean(baseline_late**2))) if len(baseline_late) else 0.0
            late_change = float(np.sqrt(np.mean(change_late**2))) if len(change_late) else change
            self.status = "activity detected" if self.baseline is not None and late_change > max(4 * baseline_energy, 1e-4) else "no significant change"
            self.presence_status = "presence model not trained"
        if response.reflection_valid and self.baseline is not None:
            self.current_candidates = extract_reflection_candidates(response, self.sample_rate, self.bandwidth_hz, self.sound_speed, self.microphone_offset, direct_ringing_guard_s=self.direct_ringing_guard_s, max_path_length_m=self.max_path_length_m)
            self.tracker.update(self.current_candidates)
            self.current_shells = self.tracker.shells()
            self.shells_stale = False
            if self.current_candidates:
                self.status = f"signal valid | {len(self.current_shells)} repeatable reflection candidates | {self.status}"
            else:
                self.status = "signal valid | no accepted reflection evidence"
        else:
            self.current_candidates = []
        self.last_response = response
        self._draw(response, difference)

    def _invalidate_room_fit(self):
        self.tracker.reset()
        self.current_candidates = []
        self.current_shells = []

    def _retain_or_expire_shells(self):
        self.tracker.update([])
        self.current_shells = self.tracker.shells()
        self.shells_stale = bool(self.current_shells)

    def _draw(self, response, difference):
        self.echo_axis.clear()
        delays_ms = response.delays * 1000.0
        magnitude = np.abs(response.correlation)
        self.echo_axis.plot(delays_ms, magnitude, color="tab:blue", linewidth=0.8)
        self.echo_axis.axvline(0.0, color="black", linestyle="--", linewidth=0.8, label="direct arrival / zero")
        for candidate in self.current_candidates:
            uncertainty_ms = candidate.delay_uncertainty_s * 1000.0
            self.echo_axis.axvspan((candidate.relative_delay_s * 1000.0) - uncertainty_ms, (candidate.relative_delay_s * 1000.0) + uncertainty_ms, color="tab:orange", alpha=0.16)
            self.echo_axis.axvline(candidate.relative_delay_s * 1000.0, color="tab:red", linewidth=0.8)
        for shell in self.current_shells:
            uncertainty_ms = shell.delay_uncertainty_s * 1000.0
            color = "0.45" if self.shells_stale else "tab:orange"
            self.echo_axis.axvspan((shell.relative_delay_s * 1000.0) - uncertainty_ms, (shell.relative_delay_s * 1000.0) + uncertainty_ms, color=color, alpha=0.10)
            self.echo_axis.axvline(shell.relative_delay_s * 1000.0, color=color, linewidth=1.1, linestyle="--")
        self.echo_axis.set_title("Measured echo response")
        self.echo_axis.set_xlabel("relative delay after direct arrival (ms)")
        self.echo_axis.set_ylabel("matched-filter amplitude (raw scale)")
        self.echo_axis.grid(alpha=0.2)
        self.echo_axis.legend(fontsize=7, loc="upper right")

        self.room_axis.clear()
        self.room_axis.set_title("Possible single-bounce reflection locations")
        self.room_axis.set_xlabel("x (m)")
        self.room_axis.set_ylabel("y (m)")
        self.room_axis.set_zlabel("z (m)")
        shell_paths = [shell.path_length_m for shell in self.current_shells]
        path_norm = plt.Normalize(min(shell_paths), max(shell_paths)) if shell_paths and min(shell_paths) < max(shell_paths) else None
        for shell_index, shell in enumerate(self.current_shells):
            surface = shell_surface(shell, self.microphone_offset, self.view_volume)
            if surface is None:
                continue
            color = "0.55" if self.shells_stale else plt.cm.viridis(path_norm(shell.path_length_m) if path_norm else 0.5)
            self.room_axis.plot_surface(*surface, alpha=0.12 if self.shells_stale else 0.16, linewidth=0.25, color=color)
            self.room_axis.text2D(0.02, 0.96 - shell_index * 0.05, f"shell {shell_index + 1}: {shell.path_length_m:.2f} ± {shell.path_uncertainty_m:.2f} m, repeats={shell.repeat_count}", transform=self.room_axis.transAxes, fontsize=7)
        self.room_axis.scatter([0], [0], [0], color="black", s=28, label="speaker S")
        microphone = np.asarray(self.microphone_offset, dtype=float)
        self.room_axis.scatter([microphone[0]], [microphone[1]], [microphone[2]], color="tab:green", s=28, label="microphone M")
        self.room_axis.set_xlim(*self.view_volume[0])
        self.room_axis.set_ylim(*self.view_volume[1])
        self.room_axis.set_zlim(*self.view_volume[2])
        geometry_text = "colocated approximation" if not self.microphone_offset_known else "measured microphone offset"
        self.room_axis.text2D(0.02, 0.02, geometry_text + "; color = path length, not confidence", transform=self.room_axis.transAxes, fontsize=8)
        self.room_axis.text2D(0.02, 0.05, f"max modeled path: {self.max_path_length_m:.1f} m; direct guard: {self.direct_ringing_guard_s * 1000:.0f} ms", transform=self.room_axis.transAxes, fontsize=8)
        if not self.current_shells:
            self.room_axis.text2D(0.5, 0.5, "No repeatable reflection shells\nare currently accepted", transform=self.room_axis.transAxes, ha="center", va="center")
        elif self.shells_stale:
            self.room_axis.text2D(0.5, 0.05, "stale: awaiting a valid measurement", transform=self.room_axis.transAxes, ha="center", va="bottom", fontsize=8, color="0.35")
        self.room_axis.legend(fontsize=7, loc="upper right")
        self.fig.suptitle(f"{self.status} | presence: {self.presence_status}", fontsize=10)
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

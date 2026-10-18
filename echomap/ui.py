from __future__ import annotations

from collections import deque
import matplotlib.pyplot as plt
import numpy as np

from .features import extract_features
from .geometry import estimate_rectangular_room
from .processing import baseline_difference, matched_filter, response_to_heatmap


class LiveView:
    def __init__(self, processor, reference, sample_rate: int, model=None, history_size=160):
        self.processor = processor
        self.reference = reference
        self.sample_rate = sample_rate
        self.model = model
        self.baseline = None
        self.history = deque(maxlen=history_size)
        self.last_response = None
        self.status = "calibrating"
        self.fig, (self.echo_axis, self.room_axis) = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.canvas.mpl_connect("key_press_event", self._key)
        self.fig.canvas.mpl_connect("close_event", lambda event: self.processor.stop())

    def _key(self, event):
        if event.key == "q":
            self.processor.stop()
            plt.close(self.fig)
        elif event.key == "b" and self.last_response is not None:
            self.baseline = self.last_response.correlation.copy()
            self.status = "calibrating"

    def update(self, recording: np.ndarray):
        response = matched_filter(recording, self.reference, self.sample_rate)
        difference = baseline_difference(response.correlation, self.baseline)
        feature_vector = extract_features(response, difference, self.sample_rate)
        if response.unreliable:
            self.status = "uncertain"
        elif self.model is not None:
            label, _, reliable = self.model.predict(feature_vector)
            self.status = "person detected" if reliable and "person" in label.lower() else (label if reliable else "uncertain")
        else:
            change = float(np.sqrt(np.mean(difference**2))) if len(difference) else 0.0
            baseline_energy = float(np.sqrt(np.mean(self.baseline**2))) if self.baseline is not None and len(self.baseline) else 0.0
            self.status = "activity detected" if self.baseline is not None and change > max(4 * baseline_energy, 1e-4) else "no significant change"
        self.last_response = response
        self.history.append(response_to_heatmap(response))
        self._draw(response)

    def _draw(self, response):
        self.echo_axis.clear()
        history = np.asarray(self.history)
        self.echo_axis.imshow(history, aspect="auto", origin="lower", interpolation="nearest", extent=[0, len(history[0]) / self.sample_rate if len(history) else 1, 0, len(history)])
        self.echo_axis.set(title=f"Echo response — {self.status}", xlabel="lag after capture (s)", ylabel="measurement")
        self.room_axis.clear()
        estimate = estimate_rectangular_room(response.delays[np.argsort(np.abs(response.correlation))[-6:]], timing_uncertainty_s=0.002)
        if estimate.supported:
            width = sum(estimate.room_width_m) / 2
            length = sum(estimate.room_length_m) / 2
            self.room_axis.plot([0, width, width, 0, 0], [0, 0, length, length, 0], "r--")
            self.room_axis.text(0.03, 0.95, "experimental estimate\nuncertain; not unique", transform=self.room_axis.transAxes, va="top")
            self.room_axis.set_xlim(0, max(width * 1.3, 1))
            self.room_axis.set_ylim(0, max(length * 1.3, 1))
        else:
            self.room_axis.text(0.5, 0.5, "Insufficient evidence\nfor room geometry", ha="center", va="center")
        self.room_axis.set(title="Estimated room view", xlabel="no measured direction (m)", ylabel="no measured direction (m)")
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

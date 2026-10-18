from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock

import numpy as np
from scipy import signal


class DiagnosticRecorder:
    """Preserves raw full-duplex blocks and derived decisions for one session."""

    def __init__(self, sample_rate: int, reference: np.ndarray, output_directory: str = "diagnostics"):
        self.sample_rate = sample_rate
        self.reference = np.asarray(reference, dtype=np.float32).copy()
        self.output_directory = Path(output_directory)
        self.microphone_blocks = []
        self.transmitted_blocks = []
        self.timestamps = []
        self.metadata = []
        self.decisions = []
        self.correlations = []
        self.correlation_delays = []
        self._lock = Lock()

    def add_audio_block(self, microphone: np.ndarray, metadata: dict) -> None:
        with self._lock:
            self.microphone_blocks.append(np.asarray(microphone, dtype=np.float32).copy())
            transmitted = metadata.get("output_block")
            if transmitted is not None:
                self.transmitted_blocks.append(np.asarray(transmitted, dtype=np.float32).copy())
            self.timestamps.append(float(metadata.get("timestamp", 0.0)))
            self.metadata.append({key: value for key, value in metadata.items() if key != "output_block"})

    def add_decision(self, response, status: str) -> None:
        peaks, _ = signal.find_peaks(np.abs(response.correlation), distance=max(1, int(0.002 * len(response.correlation))))
        selected = peaks[np.argsort(np.abs(response.correlation[peaks]))[-8:]] if len(peaks) else np.array([], dtype=int)
        self.correlations.append(np.asarray(response.correlation, dtype=np.float32).copy())
        self.correlation_delays.append(np.asarray(response.delays, dtype=np.float32).copy())
        self.decisions.append({"status": status, "aligned_delay_s": response.aligned_delay_s, "raw_direct_index": response.raw_direct_index, "direct_snr": response.direct_snr, "late_snr": response.late_snr, "reflection_ratio": response.reflection_ratio, "reflection_valid": response.reflection_valid, "reflection_reason": response.reflection_reason, "selected_peak_delays_s": response.delays[selected].tolist()})

    def save(self, prefix: str = "session") -> Path | None:
        if not self.microphone_blocks:
            return None
        self.output_directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.output_directory / f"{prefix}-{stamp}.npz"
        microphone = np.concatenate(self.microphone_blocks)
        transmitted = np.concatenate(self.transmitted_blocks) if self.transmitted_blocks else np.zeros(0, dtype=np.float32)
        correlation_width = max((len(item) for item in self.correlations), default=0)
        correlation_output = np.full((len(self.correlations), correlation_width), np.nan, dtype=np.float32)
        correlation_delays = np.full((len(self.correlation_delays), correlation_width), np.nan, dtype=np.float32)
        for index, (correlation, delays) in enumerate(zip(self.correlations, self.correlation_delays)):
            correlation_output[index, :len(correlation)] = correlation
            correlation_delays[index, :len(delays)] = delays
        metadata = {"sample_rate": self.sample_rate, "created_utc": stamp, "block_timestamps": self.timestamps, "callback_metadata": self.metadata, "decisions": self.decisions}
        np.savez_compressed(path, transmitted_waveform=self.reference, played_output=transmitted, raw_microphone=microphone, timestamps=np.asarray(self.timestamps), correlation_output=correlation_output, correlation_delays_s=correlation_delays, metadata_json=json.dumps(metadata))
        return path

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Callable

import numpy as np


@dataclass
class AudioConfig:
    sample_rate: int = 48000
    chirp_duration: float = 0.08
    repetition_interval: float = 0.5
    start_hz: float = 800.0
    end_hz: float = 8000.0
    output_level: float = 0.08
    capture_seconds: float = 0.45
    input_channel: int = 0
    output_channel: int = 0
    input_device: int | None = None
    output_device: int | None = None


def list_devices() -> list[dict]:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("sounddevice is not installed") from exc
    return [dict(device) for device in sd.query_devices()]


def device_report() -> str:
    devices = list_devices()
    lines = []
    for index, device in enumerate(devices):
        lines.append(f"{index}: {device['name']} (inputs={device['max_input_channels']}, outputs={device['max_output_channels']}, default_rate={device['default_samplerate']})")
    return "\n".join(lines)


def validate_devices(input_device: int, output_device: int, sample_rate: int, input_channel: int = 0, output_channel: int = 0) -> str:
    try:
        import sounddevice as sd
        sd.check_input_settings(device=input_device, samplerate=sample_rate, channels=input_channel + 1)
        sd.check_output_settings(device=output_device, samplerate=sample_rate, channels=output_channel + 1)
    except Exception as exc:
        raise ValueError(f"Selected devices do not support this configuration: {exc}") from exc
    devices = list_devices()
    input_name = devices[input_device]["name"]
    output_name = devices[output_device]["name"]
    return f"Confirmed microphone: {input_name}\nConfirmed speaker: {output_name}\nSample rate: {sample_rate} Hz"


class AudioEngine:
    def __init__(self, config: AudioConfig, burst: np.ndarray, callback: Callable[[np.ndarray, dict], None]):
        self.config = config
        self.burst = burst
        self.callback = callback
        self.stream = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._play_index = 0
        self._cycle_index = 0
        self._recorded: list[np.ndarray] = []
        self._metadata: list[dict] = []

    def start(self) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("Install sounddevice for live mode, or use --simulate") from exc
        channels_in = self.config.input_channel + 1
        channels_out = self.config.output_channel + 1
        self._stop.clear()
        self._play_index = 0
        self._cycle_index = 0
        self.stream = sd.Stream(
            samplerate=self.config.sample_rate,
            blocksize=0,
            device=(self.config.input_device, self.config.output_device),
            channels=(channels_in, channels_out),
            dtype="float32",
            callback=self._callback,
        )
        self.stream.start()

    def _callback(self, input_data, output_data, frames, time_info, status) -> None:
        output_data.fill(0)
        with self._lock:
            period = max(len(self.burst) + 1, int(self.config.repetition_interval * self.config.sample_rate))
            for frame in range(frames):
                phase = self._cycle_index % period
                if phase < len(self.burst):
                    output_data[frame, self.config.output_channel] = self.burst[phase]
                self._cycle_index += 1
        recorded = np.asarray(input_data[:, self.config.input_channel], dtype=np.float32).copy()
        self._recorded.append(recorded)
        self._metadata.append({"status": str(status), "input_peak": float(np.max(np.abs(recorded), initial=0.0))})
        if self.callback:
            self.callback(recorded, self._metadata[-1])

    def stop(self) -> None:
        self._stop.set()
        stream, self.stream = self.stream, None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stop()


def query_supported_rates(device: int | None = None, candidates=(16000, 32000, 44100, 48000, 96000)) -> dict[int, bool]:
    try:
        import sounddevice as sd
    except ImportError:
        return {rate: False for rate in candidates}
    result = {}
    for rate in candidates:
        try:
            sd.check_input_settings(device=device, samplerate=rate, channels=1)
            result[rate] = True
        except Exception:
            result[rate] = False
    return result

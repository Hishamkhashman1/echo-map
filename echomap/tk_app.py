from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from collections import deque
from pathlib import Path
import time

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .audio import AudioConfig, AudioEngine, list_devices, validate_devices
from .diagnostics import DiagnosticRecorder
from .signals import make_burst
from .ui import LiveView
from .model import PresenceModel


class ControlWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Echo Map — acoustic sensing")
        self.root.resizable(False, False)
        self.devices = []
        self.engine = None
        self.processor = None
        self.live_window = None
        self.live_view = None
        self.confirmed = False
        self.values = {}
        self._build()
        self.refresh_devices()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build(self):
        frame = ttk.Frame(self.root, padding=12)
        frame.grid(sticky="nsew")
        ttk.Label(frame, text="Echo Map", font=("TkDefaultFont", 16, "bold")).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(frame, text="Confirm the real microphone and speaker, then start sensing.").grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 10))
        self._combo(frame, 2, "Microphone", "input_device")
        self._combo(frame, 3, "Speaker", "output_device")
        self.advanced_visible = False
        self.advanced_frame = ttk.Frame(frame)
        ttk.Button(frame, text="Advanced settings ▸", command=self.toggle_advanced).grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 2))
        settings = [("Sample rate", "sample_rate", "48000"), ("Start frequency (Hz)", "start_hz", "800"), ("End frequency (Hz)", "end_hz", "8000"), ("Chirp duration (s)", "duration", "0.03"), ("Repeat interval (s)", "interval", "0.5"), ("Output level (0–0.25)", "level", "0.08"), ("Capture window (s)", "capture_seconds", "0.45"), ("Sound speed (m/s)", "sound_speed", "340"), ("Microphone offset x (m)", "mic_x", "0"), ("Microphone offset y (m)", "mic_y", "0"), ("Microphone offset z (m)", "mic_z", "0"), ("View half-extent (m)", "view_extent", "5")]
        for row, (label, key, default) in enumerate(settings):
            self._entry(self.advanced_frame, row, label, key, default)
        self.values["colocated"] = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.advanced_frame, text="Microphone offset unknown: use colocated spherical approximation", variable=self.values["colocated"]).grid(row=len(settings), column=0, columnspan=2, sticky="w", pady=(4, 2))

        buttons = ttk.Frame(frame)
        buttons.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(buttons, text="Refresh devices", command=self.refresh_devices).grid(row=0, column=0, padx=(0, 5))
        self.confirm_button = ttk.Button(buttons, text="Confirm devices", command=self.confirm_devices)
        self.confirm_button.grid(row=0, column=1, padx=5)
        self.start_button = ttk.Button(buttons, text="Start sensing", command=self.start)
        self.start_button.grid(row=0, column=2, padx=5)
        ttk.Button(buttons, text="Test signal", command=self.test_signal).grid(row=0, column=3, padx=5)
        ttk.Button(buttons, text="Stop", command=self.stop).grid(row=0, column=4, padx=(5, 0))
        self.status = tk.StringVar(value="Choose devices.")
        ttk.Label(frame, textvariable=self.status, wraplength=560).grid(row=7, column=0, columnspan=2, sticky="w", pady=(10, 0))

    def toggle_advanced(self):
        self.advanced_visible = not self.advanced_visible
        if self.advanced_visible:
            self.advanced_frame.grid(row=5, column=0, columnspan=2, sticky="ew")
            self.status.set("Advanced settings shown.")
        else:
            self.advanced_frame.grid_remove()
            self.status.set("Advanced settings hidden; defaults are active.")

    def _entry(self, parent, row, label, key, default):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=2)
        value = tk.StringVar(value=default)
        self.values[key] = value
        ttk.Entry(parent, textvariable=value, width=28).grid(row=row, column=1, sticky="ew", pady=2)

    def _combo(self, parent, row, label, key):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=2)
        value = tk.StringVar()
        self.values[key] = value
        combo = ttk.Combobox(parent, textvariable=value, values=[], state="readonly", width=48)
        combo.grid(row=row, column=1, sticky="ew", pady=2)
        combo.bind("<<ComboboxSelected>>", lambda event: self.unconfirm())
        self.values[f"{key}_widget"] = combo

    def refresh_devices(self):
        try:
            self.devices = list_devices()
        except RuntimeError as exc:
            self.status.set(f"Could not access audio devices: {exc}")
            return
        inputs = [f"{index}: {device['name']}" for index, device in enumerate(self.devices) if device["max_input_channels"] >= 1]
        outputs = [f"{index}: {device['name']}" for index, device in enumerate(self.devices) if device["max_output_channels"] >= 1]
        for key, values in (("input_device", inputs), ("output_device", outputs)):
            combo = self.values[f"{key}_widget"]
            combo["values"] = values
            if values:
                self.values[key].set(values[0])
        self.unconfirm()
        self.status.set(f"Found {len(inputs)} microphone input(s) and {len(outputs)} speaker output(s). Select the pair to use.")

    def unconfirm(self):
        self.confirmed = False
        if self.engine is not None:
            self.stop()

    def confirm_devices(self):
        try:
            input_device = self._device_index("input_device")
            output_device = self._device_index("output_device")
            if input_device is None or output_device is None:
                raise ValueError("Select both a microphone and a speaker")
            report = validate_devices(input_device, output_device, int(self.values["sample_rate"].get()))
        except (ValueError, IndexError) as exc:
            self.unconfirm()
            messagebox.showerror("Device check failed", str(exc))
            return
        self.confirmed = True
        self.status.set(report + "\nDevices confirmed. Press Start sensing.")

    def _device_index(self, key):
        value = self.values[key].get()
        return int(value.split(":", 1)[0]) if value else None

    def build_config(self):
        colocated = bool(self.values["colocated"].get())
        return AudioConfig(sample_rate=int(self.values["sample_rate"].get()), start_hz=float(self.values["start_hz"].get()), end_hz=float(self.values["end_hz"].get()), chirp_duration=float(self.values["duration"].get()), repetition_interval=float(self.values["interval"].get()), output_level=float(self.values["level"].get()), capture_seconds=float(self.values["capture_seconds"].get()), input_device=self._device_index("input_device"), output_device=self._device_index("output_device"), sound_speed=float(self.values["sound_speed"].get()), microphone_offset=(float(self.values["mic_x"].get()), float(self.values["mic_y"].get()), float(self.values["mic_z"].get())), microphone_offset_known=not colocated, view_half_extent_m=float(self.values["view_extent"].get()))

    def test_signal(self):
        if not self.confirmed:
            self.confirm_devices()
            if not self.confirmed:
                return
        try:
            config = self.build_config()
            reference = make_burst(config.sample_rate, config.chirp_duration, config.start_hz, config.end_hz, config.output_level)
            engine = AudioEngine(config, reference, lambda chunk, metadata: None)
            engine.start()
            self.status.set("Test signal playing through the selected speaker for one chirp.")
            self.root.after(max(100, int(config.chirp_duration * 1000 + 100)), engine.stop)
        except Exception as exc:
            messagebox.showerror("Test signal failed", str(exc))

    def start(self):
        if not self.confirmed:
            self.confirm_devices()
            if not self.confirmed:
                return
        if self.engine is not None:
            self.status.set("A sensing session is already running.")
            return
        try:
            config = self.build_config()
            reference = make_burst(config.sample_rate, config.chirp_duration, config.start_hz, config.end_hz, config.output_level)
            self.live_window = tk.Toplevel(self.root)
            self.live_window.title("Echo Map — live sensing")
            figure = Figure(figsize=(12, 5), dpi=90)
            processor = TkAudioProcessor(self.root, config.capture_seconds, self.status.set, config.sample_rate, reference)
            self.processor = processor
            model = None
            if Path("echo-model.pkl").exists():
                try:
                    model = PresenceModel.load("echo-model.pkl")
                except Exception:
                    model = None
            view_volume = ((-config.view_half_extent_m, config.view_half_extent_m), (-config.view_half_extent_m, config.view_half_extent_m), (-config.view_half_extent_m, config.view_half_extent_m))
            self.live_view = LiveView(processor, reference, config.sample_rate, model=model, figure=figure, bandwidth_hz=abs(config.end_hz - config.start_hz), sound_speed=config.sound_speed, microphone_offset=config.microphone_offset, microphone_offset_known=config.microphone_offset_known, view_volume=view_volume)
            processor.view = self.live_view
            canvas = FigureCanvasTkAgg(figure, master=self.live_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.live_window.protocol("WM_DELETE_WINDOW", self.stop)
            self.engine = AudioEngine(config, reference, processor.accept)
            processor.engine = self.engine
            self.engine.start()
            processor.schedule()
        except Exception as exc:
            self.stop()
            messagebox.showerror("Could not start sensing", str(exc))
            return
        self.status.set("Sensing started. Saving raw diagnostics and checking signal validity...")

    def stop(self):
        if self.engine is not None:
            self.engine.stop()
            self.engine = None
        diagnostic_path = None
        if self.processor is not None:
            diagnostic_path = self.processor.stop()
            self.processor = None
        if self.live_view is not None:
            self.live_view.close()
            self.live_view = None
        if self.live_window is not None:
            self.live_window.destroy()
            self.live_window = None
        self.status.set(f"Sensing stopped. Diagnostics saved to {diagnostic_path}." if diagnostic_path else "Sensing stopped.")

    def close(self):
        self.stop()
        self.root.destroy()


class TkAudioProcessor:
    def __init__(self, root, capture_seconds, status_callback, sample_rate, reference):
        self.root = root
        self.capture_seconds = capture_seconds
        self.status_callback = status_callback
        self.diagnostics = DiagnosticRecorder(sample_rate, reference)
        self.probe_buffer = np.empty(0, dtype=np.float32)
        self.active_probe_index = None
        self.pending_recordings = deque()
        self.view = None
        self.engine = None

    def accept(self, chunk, metadata):
        chunk = np.asarray(chunk, dtype=np.float32)
        probe_index = metadata.get("probe_index")
        start_offset = metadata.get("probe_start_offset")
        if self.active_probe_index is None:
            self.active_probe_index = probe_index
            if start_offset is not None:
                self.probe_buffer = chunk[int(start_offset):].copy()
        elif start_offset is not None:
            offset = int(start_offset)
            if offset:
                self.probe_buffer = np.concatenate((self.probe_buffer, chunk[:offset]))
            if len(self.probe_buffer):
                self.pending_recordings.append(self.probe_buffer.copy())
            self.active_probe_index = metadata.get("next_probe_index", probe_index)
            self.probe_buffer = chunk[offset:].copy()
        elif probe_index != self.active_probe_index:
            if len(self.probe_buffer):
                self.pending_recordings.append(self.probe_buffer.copy())
            self.active_probe_index = probe_index
            self.probe_buffer = chunk[int(start_offset):].copy() if start_offset is not None else np.empty(0, dtype=np.float32)
        else:
            self.probe_buffer = np.concatenate((self.probe_buffer, chunk))
        self.diagnostics.add_audio_block(chunk, metadata)

    def schedule(self):
        if self.view is None:
            return
        needed = max(1, int(self.capture_seconds * self.view.sample_rate))
        if self.pending_recordings:
            buffered = self.pending_recordings.popleft()
            if len(buffered) >= needed:
                recording = buffered[:needed]
            else:
                recording = None
            if recording is not None:
                self.view.update(recording)
                self.diagnostics.add_decision(self.view.last_response, self.view.status)
                self.status_callback(self.status_text())
        self.root.after(50, self.schedule)

    def status_text(self):
        if self.view.baseline is None:
            return f"Calibrating startup reference: {len(self.view.calibration_samples)}/{self.view.calibration_measurements}. Keep the laptop and scene still."
        if self.view.current_shells:
            shell_text = f"{len(self.view.current_shells)} repeatable reflection shell(s)"
        else:
            shell_text = "no repeatable reflection shells"
        return f"Live sensing: measurement {self.view.measurement_index}; {shell_text}. {self.view.status}. Probes repeat automatically; shells accumulate across valid repeats. Timing uncertainty is shown on the echo plot."

    def stop(self):
        if self.engine is not None:
            self.engine.stop()
        return self.diagnostics.save()


def launch():
    root = tk.Tk()
    ControlWindow(root)
    root.mainloop()

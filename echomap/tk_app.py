from __future__ import annotations

import subprocess
import sys
import tkinter as tk
from tkinter import messagebox, ttk

from .audio import list_devices, validate_devices


class ControlWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Echo Map — acoustic sensing")
        self.root.resizable(False, False)
        self.devices = []
        self.process = None
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
        settings = [("Sample rate", "sample_rate", "48000"), ("Start frequency (Hz)", "start_hz", "800"), ("End frequency (Hz)", "end_hz", "8000"), ("Chirp duration (s)", "duration", "0.08"), ("Repeat interval (s)", "interval", "0.5"), ("Output level (0–0.25)", "level", "0.08"), ("Capture window (s)", "capture_seconds", "0.45")]
        for row, (label, key, default) in enumerate(settings, 4):
            self._entry(frame, row, label, key, default)

        buttons = ttk.Frame(frame)
        buttons.grid(row=11, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(buttons, text="Refresh devices", command=self.refresh_devices).grid(row=0, column=0, padx=(0, 5))
        self.confirm_button = ttk.Button(buttons, text="Confirm devices", command=self.confirm_devices)
        self.confirm_button.grid(row=0, column=1, padx=5)
        self.start_button = ttk.Button(buttons, text="Start sensing", command=self.start, state="disabled")
        self.start_button.grid(row=0, column=2, padx=5)
        ttk.Button(buttons, text="Stop", command=self.stop).grid(row=0, column=3, padx=(5, 0))
        self.status = tk.StringVar(value="Choose devices.")
        ttk.Label(frame, textvariable=self.status, wraplength=560).grid(row=12, column=0, columnspan=2, sticky="w", pady=(10, 0))

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
        if hasattr(self, "start_button"):
            self.start_button.configure(state="disabled")

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
        self.start_button.configure(state="normal")
        self.status.set(report + "\nDevices confirmed. Press Start sensing.")

    def _device_index(self, key):
        value = self.values[key].get()
        return int(value.split(":", 1)[0]) if value else None

    def command(self):
        command = [sys.executable, "-m", "echomap.cli", "--headless", "--input-device", str(self._device_index("input_device")), "--output-device", str(self._device_index("output_device"))]
        for key, option in [("sample_rate", "sample-rate"), ("start_hz", "start-hz"), ("end_hz", "end-hz"), ("duration", "duration"), ("interval", "interval"), ("level", "level"), ("capture_seconds", "capture-seconds")]:
            command.extend([f"--{option}", self.values[key].get()])
        return command

    def start(self):
        if not self.confirmed:
            return
        if self.process and self.process.poll() is None:
            self.status.set("A sensing session is already running.")
            return
        try:
            command = self.command()
        except ValueError as exc:
            messagebox.showerror("Cannot start", str(exc))
            return
        self.process = subprocess.Popen(command)
        self.status.set("Sensing started. Use B for baseline and Q in the Matplotlib window to stop.")

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.status.set("Stopping sensing…")
        else:
            self.status.set("No sensing session is running.")

    def close(self):
        self.stop()
        self.root.destroy()


def launch():
    root = tk.Tk()
    ControlWindow(root)
    root.mainloop()

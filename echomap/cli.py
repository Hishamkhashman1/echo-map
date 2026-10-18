from __future__ import annotations

import argparse
import time
import uuid
import numpy as np

from .audio import AudioConfig, AudioEngine, device_report, query_supported_rates
from .features import extract_features
from .model import PresenceModel, TrainingExample, load_examples, save_examples
from .processing import matched_filter
from .signals import make_burst
from .simulation import synthetic_recording
from .ui import LiveView


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Stationary single-speaker/single-microphone acoustic sensing experiment")
    result.add_argument("--simulate", action="store_true", help="run with synthetic echoes and no audio hardware")
    result.add_argument("--list-devices", action="store_true")
    result.add_argument("--input-device", type=int)
    result.add_argument("--output-device", type=int)
    result.add_argument("--sample-rate", type=int, default=48000)
    result.add_argument("--start-hz", type=float, default=800)
    result.add_argument("--end-hz", type=float, default=8000)
    result.add_argument("--duration", type=float, default=0.08)
    result.add_argument("--interval", type=float, default=0.5)
    result.add_argument("--level", type=float, default=0.08)
    result.add_argument("--capture-seconds", type=float, default=0.45)
    result.add_argument("--model")
    result.add_argument("--train")
    result.add_argument("--record-label", help="save live feature vectors under this training label")
    result.add_argument("--examples", default="training-examples.json", help="JSON file for labeled feature rows")
    result.add_argument("--session", default=None, help="recording-session id; defaults to a new id")
    return result


def run_training(path: str, model_path: str = "echo-model.pkl") -> None:
    model = PresenceModel()
    report = model.fit(load_examples(path), {"sample_rate": 48000})
    model.save(model_path)
    print(report)
    print(f"Saved {model_path}; evaluation uses a held-out recording session.")


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    if args.list_devices:
        print(device_report())
        print("Supported rates:", query_supported_rates(args.input_device))
        return
    if args.train:
        run_training(args.train)
        return
    config = AudioConfig(sample_rate=args.sample_rate, chirp_duration=args.duration, repetition_interval=args.interval, start_hz=args.start_hz, end_hz=args.end_hz, output_level=min(max(args.level, 0.0), 0.25), capture_seconds=args.capture_seconds, input_device=args.input_device, output_device=args.output_device)
    reference = make_burst(config.sample_rate, config.chirp_duration, config.start_hz, config.end_hz, config.output_level)
    model = PresenceModel.load(args.model) if args.model else None
    if args.simulate:
        view = LiveView(_SimulationProcessor(view=None), reference, config.sample_rate, model=model)
        view.processor.view = view
        for index in range(100):
            view.update(synthetic_recording(reference, config.sample_rate, movement=0.006 if 35 < index < 65 else 0.0, seed=index))
            plt_pause()
            if not plt_is_open(view.fig):
                break
        return
    processor = _LiveProcessor(label=args.record_label, examples_path=args.examples, session=args.session)
    view = LiveView(processor, reference, config.sample_rate, model=model)
    view.processor.view = view
    engine = AudioEngine(config, reference, lambda chunk, metadata: view.processor.accept(chunk))
    view.processor.engine = engine
    try:
        engine.start()
        print("Running. Press B to capture baseline; Q or close the window to stop.")
        while plt_is_open(view.fig):
            plt_pause(0.05)
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        processor.save()


def plt_pause(seconds):
    import matplotlib.pyplot as plt
    plt.pause(seconds)


def plt_is_open(fig):
    return bool(plt_pause_check(fig))


def plt_pause_check(fig):
    import matplotlib.pyplot as plt
    return plt.fignum_exists(fig.number)


class _SimulationProcessor:
    def __init__(self, view):
        self.view = view
    def stop(self):
        pass
    def accept(self, chunk):
        self.view.update(chunk)


class _LiveProcessor(_SimulationProcessor):
    def __init__(self, label=None, examples_path="training-examples.json", session=None):
        super().__init__(None)
        self.engine = None
        self._chunks = []
        self._last = time.monotonic()
        self.label = label
        self.examples_path = examples_path
        self.session = session or uuid.uuid4().hex[:12]
        self.examples = []
    def accept(self, chunk):
        self._chunks.append(chunk)
        if time.monotonic() - self._last >= 0.45:
            recording = np.concatenate(self._chunks)
            self._chunks = []
            self._last = time.monotonic()
            self.view.update(recording)
            if self.label and self.view.last_response is not None:
                difference = self.view.last_response.correlation if self.view.baseline is None else self.view.last_response.correlation - self.view.baseline
                self.examples.append(TrainingExample(extract_features(self.view.last_response, difference, self.view.sample_rate), self.label, self.session))

    def save(self):
        if not self.examples:
            return
        try:
            existing = load_examples(self.examples_path)
        except FileNotFoundError:
            existing = []
        save_examples(self.examples_path, existing + self.examples)
        print(f"Saved {len(self.examples)} labeled examples to {self.examples_path} (session {self.session}).")
    def stop(self):
        if self.engine is not None:
            self.engine.stop()


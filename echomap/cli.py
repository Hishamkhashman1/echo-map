from __future__ import annotations

import argparse
import time
import uuid
import numpy as np

from .audio import AudioConfig, AudioEngine, device_report, query_supported_rates
from .features import aggregate_sequence_features, extract_features
from .model import PresenceModel, TrainingExample, load_examples, save_examples
from .processing import matched_filter
from .room_inference import RoomCandidate, fit_room_candidates, simulate_room_response
from .signals import make_burst
from .simulation import synthetic_recording
from .ui import LiveView


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Stationary single-speaker/single-microphone acoustic sensing experiment")
    result.add_argument("--gui", action="store_true", help="open the Tkinter control window")
    result.add_argument("--headless", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--simulate", action="store_true", help="run with synthetic echoes and no audio hardware")
    result.add_argument("--list-devices", action="store_true")
    result.add_argument("--input-device", type=int)
    result.add_argument("--output-device", type=int)
    result.add_argument("--sample-rate", type=int, default=48000)
    result.add_argument("--start-hz", type=float, default=800)
    result.add_argument("--end-hz", type=float, default=8000)
    result.add_argument("--duration", type=float, default=0.03)
    result.add_argument("--interval", type=float, default=0.5)
    result.add_argument("--level", type=float, default=0.08)
    result.add_argument("--capture-seconds", type=float, default=0.45)
    result.add_argument("--model")
    result.add_argument("--train")
    result.add_argument("--room-experiment", action="store_true", help="fit ranked room candidates to a synthetic labeled room")
    result.add_argument("--room-evaluate", action="store_true", help="evaluate room fitting on held-out synthetic rooms")
    result.add_argument("--room-backend", choices=("auto", "analytic", "pyroomacoustics"), default="auto")
    result.add_argument("--plot-diagnostics", help="plot a saved diagnostic .npz session")
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


def run_room_experiment(sample_rate: int, backend: str) -> None:
    reference = make_burst(sample_rate, 0.04, 500, 3500, 0.5)
    known = RoomCandidate(5.0, 6.0, 2.5, (2.5, 3.0), (2.62, 3.0), 0.6)
    measured, used = simulate_room_response(reference, sample_rate, known, 0.45, backend=backend)
    result = fit_room_candidates(measured, sample_rate, reference, backend=backend)
    print(f"SYNTHETIC ONLY — backend={used}, known room={known.width_m}m x {known.length_m}m")
    print(result.explanation)
    for index, candidate in enumerate(result.candidates, 1):
        print(f"{index}. {candidate.width_m:.1f}m x {candidate.length_m:.1f}m, device=({candidate.speaker_xy[0]:.2f},{candidate.speaker_xy[1]:.2f}), absorption={candidate.absorption:.2f}, score={candidate.score:.4f}")


def run_room_evaluation(sample_rate: int, backend: str) -> None:
    reference = make_burst(sample_rate, 0.04, 500, 3500, 0.5)
    held_out = [RoomCandidate(3.5, 4.5, 2.5, (1.75, 2.25), (1.87, 2.25), 0.2), RoomCandidate(6.5, 7.5, 2.5, (4.23, 3.75), (4.35, 3.75), 0.6)]
    errors = []
    for known in held_out:
        measured, used = simulate_room_response(reference, sample_rate, known, 0.45, backend=backend)
        result = fit_room_candidates(measured, sample_rate, reference, backend=backend)
        best = result.best
        if best is None:
            continue
        error = float(np.mean([abs(best.width_m - known.width_m), abs(best.length_m - known.length_m)]))
        errors.append(error)
        print(f"held-out {known.width_m:.1f} x {known.length_m:.1f} m -> top candidate {best.width_m:.1f} x {best.length_m:.1f} m, rank-1 score={best.score:.4f}")
    print(f"SYNTHETIC EVALUATION ONLY — backend={used}, dimension MAE={np.mean(errors):.3f} m across {len(errors)} held-out rooms")


def plot_diagnostics(path: str) -> None:
    import json
    import matplotlib.pyplot as plt

    values = np.load(path)
    metadata = json.loads(str(values["metadata_json"]))
    figure, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(values["raw_microphone"], linewidth=0.5, label="raw microphone")
    if len(values["played_output"]):
        axes[0].plot(values["played_output"], linewidth=0.5, alpha=0.7, label="actual output blocks")
    axes[0].set_title("Raw diagnostic levels — not normalized")
    axes[0].legend()
    correlation = values["correlation_output"]
    if correlation.size:
        axes[1].imshow(np.abs(correlation), aspect="auto", origin="lower", interpolation="nearest")
    axes[1].set_title("Matched-filter correlation magnitude")
    axes[1].set_xlabel("relative delay sample")
    axes[1].set_ylabel("measurement")
    print(json.dumps(metadata["decisions"][-5:], indent=2))
    figure.tight_layout()
    plt.show()


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    if args.gui or (argv is None and not any(option.startswith("--") for option in __import__("sys").argv[1:])):
        from .tk_app import launch
        launch()
        return
    if args.list_devices:
        print(device_report())
        print("Supported rates:", query_supported_rates(args.input_device))
        return
    if args.train:
        run_training(args.train)
        return
    if args.room_experiment:
        run_room_experiment(args.sample_rate, args.room_backend)
        return
    if args.room_evaluate:
        run_room_evaluation(args.sample_rate, args.room_backend)
        return
    if args.plot_diagnostics:
        plot_diagnostics(args.plot_diagnostics)
        return
    config = AudioConfig(sample_rate=args.sample_rate, chirp_duration=args.duration, repetition_interval=args.interval, start_hz=args.start_hz, end_hz=args.end_hz, output_level=min(max(args.level, 0.0), 0.25), capture_seconds=args.capture_seconds, input_device=args.input_device, output_device=args.output_device)
    reference = make_burst(config.sample_rate, config.chirp_duration, config.start_hz, config.end_hz, config.output_level)
    model = PresenceModel.load(args.model) if args.model else None
    if args.simulate:
        view = LiveView(_SimulationProcessor(view=None), reference, config.sample_rate, model=model, bandwidth_hz=abs(config.end_hz - config.start_hz), sound_speed=config.sound_speed, microphone_offset=config.microphone_offset, microphone_offset_known=config.microphone_offset_known, view_volume=((-config.view_half_extent_m, config.view_half_extent_m),) * 2 + ((-config.view_half_extent_m, config.view_half_extent_m),))
        view.processor.view = view
        for index in range(100):
            view.update(synthetic_recording(reference, config.sample_rate, movement=0.006 if 35 < index < 65 else 0.0, seed=index))
            plt_pause(0.01)
            if not plt_is_open(view.fig):
                break
        return
    processor = _LiveProcessor(label=args.record_label, examples_path=args.examples, session=args.session)
    view = LiveView(processor, reference, config.sample_rate, model=model, bandwidth_hz=abs(config.end_hz - config.start_hz), sound_speed=config.sound_speed, microphone_offset=config.microphone_offset, microphone_offset_known=config.microphone_offset_known, view_volume=((-config.view_half_extent_m, config.view_half_extent_m),) * 2 + ((-config.view_half_extent_m, config.view_half_extent_m),))
    view.processor.view = view
    engine = AudioEngine(config, reference, lambda chunk, metadata: view.processor.accept(chunk))
    view.processor.engine = engine
    try:
        engine.start()
        import matplotlib.pyplot as plt
        plt.show(block=False)
        print("Running. Automatic baseline calibration is in progress; close the window or press Q to stop.")
        while plt_is_open(view.fig):
            plt_pause(0.05)
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        processor.save()
        view.close()


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
        self.frame_features = []
    def accept(self, chunk):
        self._chunks.append(chunk)
        if time.monotonic() - self._last >= 0.45:
            recording = np.concatenate(self._chunks)
            self._chunks = []
            self._last = time.monotonic()
            self.view.update(recording)
            if self.label and self.view.last_response is not None:
                difference = self.view.last_response.correlation if self.view.baseline is None else self.view.last_response.correlation - self.view.baseline
                self.frame_features.append(extract_features(self.view.last_response, difference, self.view.sample_rate))

    def save(self):
        if self.label and self.frame_features:
            self.examples = [TrainingExample(aggregate_sequence_features(np.asarray(self.frame_features)), self.label, self.session)]
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


if __name__ == "__main__":
    main()


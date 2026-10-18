# Echo Map

Echo Map is a deliberately experimental acoustic sensing application for a stationary laptop using one speaker output and one microphone input. It emits a known audible chirp, records the microphone response, and compares aligned matched-filter responses over time. It is not a security system, medical device, occupancy guarantee, or room scanner.

The application has no web frontend. Its live view is one Matplotlib window with a scrolling echo heatmap and a cautious, explicitly labeled experimental room estimate.

## Install

Ubuntu audio development packages are commonly needed before installing Python dependencies:

```bash
sudo apt install libportaudio2 portaudio19-dev python3-venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Run

Try the software without audio hardware first:

```bash
echo-map --simulate
```

List devices and probe common input rates:

```bash
echo-map --list-devices
```

Start live sensing with conservative defaults:

```bash
echo-map --input-device 1 --output-device 1
```

Useful settings include `--sample-rate`, `--start-hz`, `--end-hz`, `--duration`, `--interval`, `--level`, and `--capture-seconds`. The initial level is 0.08 and the command clamps requested level to 0.25. Increase it only while monitoring comfort and clipping. Higher frequencies are not assumed to be ultrasonic or inaudible; the audible calibration chirp is intentional.

Press `B` after the room is quiet to capture the current empty-room baseline. Press `Q` or close the window to stop. Audio resources are closed in normal shutdown and error cleanup.

## What the signal processing means

The recorded signal is correlated with the emitted chirp. The strongest early peak is treated as the direct speaker-to-microphone path and is not interpreted as object distance. A small guard region around it is excluded only for late-reflection features; the full response remains visible. Measurements are compared at their aligned correlation lags, not at raw playback/recording timestamps.

An acoustic path length is approximately `speed_of_sound * delay`. Converting that path to object range requires a geometry assumption; a wall reflection is generally a multi-segment path, not twice the microphone-to-object range. At room temperature, 343 m/s is only an approximation. Timing uncertainty is displayed as uncertainty in path length. Range resolution is set primarily by chirp bandwidth (`c / bandwidth` for a simple matched-filter scale), not by sample spacing alone.

A single stationary speaker/microphone pair has no directional measurement. The room panel therefore shows only a tentative rectangular estimate when separated reflection delays are available, with uncertainty and a statement that alternate layouts remain possible. It does not place people, furniture, or directional walls.

## Calibration and troubleshooting

Start with an audible band such as 500–4000 Hz. Measure several bands and retain only bands with a stable direct peak and a reasonable signal-to-noise ratio. Avoid assuming a laptop speaker or microphone supports ultrasound. Keep the laptop stationary, and do not interpret the direct-path latency as a reflection.

For reproducible measurements, disable or bypass PulseAudio/PipeWire processing where your system permits it: use a raw/monitor-free device, turn off echo cancellation, automatic gain control, noise suppression, and “enhancements” in the selected input profile. The exact controls are distribution and driver dependent. The program flags weak or noisy matched-filter responses as uncertain, and status metadata includes callback warnings and input peaks.

Clipping means the level is too high or the input path is applying gain. Dropped-buffer warnings, changing gain, active echo cancellation, fan noise, and a moving laptop make comparisons unreliable. The program does not claim that failure to observe breathing or a reflection proves a person is absent.

## Labeled training data

The core runtime can load a model using `--model model.pkl`. A small training utility is included through the Python API. Each example consists of the extracted seven-feature vector, a label, and a recording-session identifier. Use labels such as `empty room`, `person moving`, `person sitting still`, and `environmental disturbance`.

Collect multiple separated sessions, not many adjacent frames from one session. Keep entire sessions together when evaluating. The live capture workflow appends extracted feature rows with a stable session id:

```bash
echo-map --record-label "empty room" --session empty-01 --examples training-examples.json
echo-map --record-label "person moving" --session moving-01 --examples training-examples.json
echo-map --record-label "person sitting still" --session still-01 --examples training-examples.json
echo-map --record-label "environmental disturbance" --session fan-01 --examples training-examples.json
echo-map --train training-examples.json
```

Press `Q` after each session. The label is saved only when the explicit `--record-label` option is supplied. A minimal workflow is:

1. Record several quiet empty-room sessions at each usable chirp band.
2. Record separate sessions with a person moving and sitting still.
3. Record fan/curtain/door disturbances as their own label.
4. Extract features from each aligned response and save rows as JSON with a stable session id.
5. Train with `PresenceModel.fit`; it holds out the final complete session, reports confusion matrix, precision, and recall, then saves the model, preprocessing, feature schema, and acquisition settings.

The model deliberately returns `uncertain` when confidence and class margin are weak. Its score is not a calibrated probability. False positives should be reported from the disturbance label and from empty-room sessions; never silently relabel environmental change as a person.

## Tests and hardware status

Run software tests with:

```bash
python -m unittest discover -s tests
```

The tests cover known synthetic delays, baseline subtraction, finite feature extraction, and explicit path-length uncertainty. `--simulate` verifies the processing and live plot without physical hardware. Physical full-duplex timing, supported rates, driver behavior, system processing, comfort, and usable frequency bands still need validation on the target laptop.

Optional `pyroomacoustics` is intentionally not required: the built-in simulator provides visibly synthetic echoes and avoids presenting a simulated room as a live measurement.

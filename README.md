# Echo Map

Echo Map is a deliberately experimental acoustic measurement prototype for a stationary laptop using one speaker output and one microphone input. It emits a known audible chirp, records the microphone response, aligns it to the direct path, validates late reflection evidence, and accumulates 3D geometric constraint shells. It is not a security system, medical device, occupancy guarantee, or arbitrary room scanner.

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

The easiest way to use the application is the Tkinter control window:

```bash
echo-map
```

It discovers input/output devices, checks that the selected microphone and speaker support the chosen sample rate, edits the core chirp settings, and starts/stops live sensing. The Matplotlib view automatically captures a startup reference from the first eight reliable measurements; this is not automatically labeled as an empty room. It then shows the measured echo response and possible single-bounce reflection shells. Close the live view or press `Q` to stop.

The same window can be opened explicitly with `echo-map --gui`.

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

Keep the laptop and scene stable for the first few seconds after starting so the startup reference is useful. It is not proof that the room is vacant. Press `Q` or close the window to stop. Audio resources are closed in normal shutdown and error cleanup.

During sensing the selected speaker emits one chirp every configured repeat interval (0.5 seconds by default); it is not a single one-time measurement. Each capture window must be shorter than that interval and contains exactly one probe response. A candidate becomes a displayed shell only after four matching observations. The live status shows the measurement number. Strong matched-filter sidelobes are capped to the six strongest late candidates, rather than being accumulated as dozens of apparent reflectors. A brief invalid capture leaves the last shells visible in gray as `stale`; restarting sensing clears the session evidence.

## What the signal processing means

The recorded signal is correlated with the emitted chirp. The strongest early peak is treated as the direct speaker-to-microphone path and is not interpreted as object distance. A small guard region around it is excluded only for late-reflection features; the full response remains visible. Measurements are compared at their aligned correlation lags, not at raw playback/recording timestamps.

The probe is a deterministic phase-coded chirp: its 13-chip sign pattern is known to the receiver and is included in the matched-filter reference. This is a signal-identification code, not encryption. It improves rejection of unrelated noise and makes the transmitted waveform inspectable, but it cannot by itself distinguish a real room reflection from repeatable speaker/microphone ringing.

An acoustic path length is approximately `speed_of_sound * delay`. Converting that path to object range requires a geometry assumption; a wall reflection is generally a multi-segment path, not twice the microphone-to-object range. At room temperature, 343 m/s is only an approximation. Timing uncertainty is displayed as uncertainty in path length. Range resolution is set primarily by chirp bandwidth (`c / bandwidth` for a simple matched-filter scale), not by sample spacing alone.

A single stationary speaker/microphone pair has no directional measurement. The live 3D panel therefore shows shells of possible reflection locations, not walls, objects, or directional points.

The live extractor excludes a short measured direct-path/ringing margin and rejects candidate paths outside the configured viewing volume. The default audible chirp is 30 ms, leaving ordinary room reflection delays observable while retaining a conservative early guard. A dense set of early peaks is treated as hardware response or insufficient evidence, not converted into a room map.

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

The tests cover known synthetic delays, baseline subtraction, finite feature extraction, sequence features, raw diagnostic preservation, zero input, unrelated noise, direct-path-only input, playback-disabled input, and ranked synthetic room fits. Physical full-duplex timing, supported rates, driver behavior, system processing, comfort, and usable frequency bands still need validation on the target laptop.

## Physical validation procedure

The application now saves every GUI sensing session under `diagnostics/`. Files preserve the reference chirp, actual output blocks, raw microphone samples, callback timestamps, correlation arrays, alignment offsets, selected peak delays, signal-validity decisions, and inference decisions. No raw signal is normalized before saving.

Run the GUI and use the same selected microphone/speaker pair for each trial:

1. **Normal speaker playback:** place the laptop stationary, start sensing, wait for startup reference capture, then leave the room unchanged for 20 seconds. Confirm `signal detected` and look for stable late-reflection peaks.
2. **Silent playback:** set output level to zero in Advanced settings and restart sensing. It must report `No reliable transmitted chirp` and must not show a room estimate.
3. **Headphones away from microphone:** play the chirp through headphones while the laptop speaker is selected, keep the headphones physically away from the laptop microphone, and restart sensing. It must report no usable late reflections; a direct electrical/airborne signal alone is not room evidence.
4. **Measured hard reflector:** place a large flat hard board at a measured distance along a repeatable path. Record one session, move it to a second measured distance, record again, and compare the saved correlation plots with `echo-map --plot-diagnostics diagnostics/<session>.npz`. A changed reflection should move by the expected change in acoustic path delay, subject to the speaker/microphone geometry; the direct path should not move.

The tests above are procedures, not claimed hardware results. The repository has no access to your room or laptop during development, so hardware validation remains pending until you run them. A device name is never used to decide whether a measurement is valid; the decision comes from measured direct-path SNR, late-reflection SNR, reflection-to-direct ratio, and the matched-filter sidelobe/noise gate.

## Physics constraint view

The live view uses `S = (0, 0, 0)` for the speaker and an Advanced-settings microphone offset `M`. With a known offset, a candidate relative reflection delay `t` gives a path length `L = |S-M| + c*t` and the possible single-bounce location set:

```text
|P-S| + |P-M| = L
```

This is an ellipsoid. If the microphone offset is unknown, the UI explicitly uses the colocated approximation and displays a spherical shell of radius `L/2`. Shell thickness is derived from chirp bandwidth, sample spacing, direct-path timing uncertainty, and repeatability; it is not artificially sharpened. The finite viewing volume is only a plotting window and is never treated as a wall. Repeated measurements increase the shell repeat count but do not create new viewpoints or angular information.

The left panel shows raw-scale matched-filter amplitude versus delay, the direct arrival at zero, candidate reflection windows, and their uncertainty. The right panel shows translucent shells labeled `possible single-bounce reflection locations`, plus speaker and microphone positions. Silence, unrelated noise, direct-only input, clipping, or a failed reflection gate clears the shells.

## Optional model-based room inference

The rectangular-room fitter remains a separate experimental command and is not used to turn the live shells into a room map. It varies rectangular width, length, absorption, and a small grid of speaker/microphone positions with a known or assumed 0.12 m separation. Candidate impulse responses use `pyroomacoustics` when installed; otherwise a clearly labeled first-order image-source approximation is used. Candidate signatures are log-magnitude, gain-normalized, direct-path-excluded responses, so the fit is less sensitive to laptop frequency response and volume.

Install the optional simulator and run the reproducible synthetic experiment:

```bash
python -m pip install -e '.[room]'
echo-map --room-experiment --room-backend pyroomacoustics
echo-map --room-evaluate --room-backend pyroomacoustics
```

The terminal labels these results `SYNTHETIC ONLY`. Synthetic success does not establish real-room accuracy; measured-room validation is still required.

## Presence model

Without a saved model the live status explicitly says `presence model not trained`; the separate activity detector only reports acoustic change. The terminal recorder aggregates aligned frame features into sequence features for each labeled session:

```bash
echo-map --record-label "empty room" --session empty-01 --examples training-examples.json
echo-map --record-label "person moving" --session moving-01 --examples training-examples.json
echo-map --record-label "person sitting still" --session still-01 --examples training-examples.json
echo-map --record-label "environmental disturbance" --session fan-01 --examples training-examples.json
echo-map --train training-examples.json
```

Use multiple complete sessions per label. The trainer holds out the final session, reports confusion matrix, precision, recall, and the held-out session id, and saves the scaler, classifier, feature schema, and acquisition settings to `echo-model.pkl`. Start live sensing with `--model echo-model.pkl` to load it; otherwise the app will not claim person detection.

The current repository has no measured-room recording set, so real-room geometry error and real presence accuracy are unvalidated. Collect recordings from rooms whose dimensions and device position are measured independently before treating any output as evidence of performance.

import numpy as np
import unittest
import tempfile

from echomap.features import extract_features
from echomap.features import aggregate_sequence_features, SEQUENCE_FEATURE_SCHEMA
from echomap.geometry import path_lengths_from_delays
from echomap.processing import baseline_difference, matched_filter
from echomap.room_inference import RoomCandidate, fit_room_candidates, simulate_room_response
from echomap.diagnostics import DiagnosticRecorder
from echomap.constraints import ReflectionCandidate, ReflectionTracker, extract_reflection_candidates, shell_surface
from echomap.signals import PROBE_PHASE_CODE, make_burst
from echomap.simulation import synthetic_recording


class PipelineTests(unittest.TestCase):
    def test_probe_uses_known_phase_code(self):
        sample_rate = 13000
        duration = 0.13
        uncoded = make_burst(sample_rate, duration, 500, 3500, 0.5)
        self.assertEqual(len(PROBE_PHASE_CODE), 13)
        self.assertEqual(len(uncoded), int(round(sample_rate * duration)))
        self.assertGreater(np.max(np.abs(uncoded)), 0.0)

    def test_matched_filter_finds_known_echo_delay(self):
        sample_rate = 16000
        reference = make_burst(sample_rate, 0.04, 500, 3500, 0.5)
        recording = synthetic_recording(reference, sample_rate, delays_s=(0.01, 0.035), gains=(1.0, 0.3), noise=0)
        response = matched_filter(recording, reference, sample_rate)
        self.assertLess(abs(response.aligned_delay_s - 0.01), 2 / sample_rate)
        self.assertEqual(response.direct_index, 0)
        self.assertAlmostEqual(response.delays[0], 0.0)

    def test_matched_filter_can_find_delayed_direct_path(self):
        sample_rate = 16000
        reference = make_burst(sample_rate, 0.04, 500, 3500, 0.5)
        recording = synthetic_recording(reference, sample_rate, delays_s=(0.20,), gains=(1.0,), noise=0)
        response = matched_filter(recording, reference, sample_rate)
        self.assertFalse(response.unreliable)
        self.assertAlmostEqual(response.aligned_delay_s, 0.20, delta=2 / sample_rate)


    def test_baseline_comparison_preserves_shape_and_detects_change(self):
        baseline = np.zeros(10)
        current = np.ones(10)
        difference = baseline_difference(current, baseline)
        self.assertEqual(difference.shape, current.shape)
        self.assertTrue(np.allclose(difference, 1))


    def test_feature_vector_is_finite_and_schema_sized(self):
        sample_rate = 16000
        reference = make_burst(sample_rate, 0.04, 500, 3500, 0.5)
        response = matched_filter(synthetic_recording(reference, sample_rate, noise=0), reference, sample_rate)
        values = extract_features(response, response.correlation, sample_rate)
        self.assertEqual(values.shape, (7,))
        self.assertTrue(np.all(np.isfinite(values)))


    def test_path_length_conversion_is_explicit(self):
        paths, uncertainty = path_lengths_from_delays(np.array([0.01]), speed_m_s=343, timing_uncertainty_s=0.001)
        self.assertTrue(np.allclose(paths, [3.43]))
        self.assertTrue(np.allclose(uncertainty, [0.343]))

    def test_sequence_features_capture_temporal_change(self):
        frames = np.arange(35, dtype=float).reshape(5, 7)
        values = aggregate_sequence_features(frames)
        self.assertEqual(values.shape, (len(SEQUENCE_FEATURE_SCHEMA),))
        self.assertGreater(values[14], 0)

    def test_room_fit_returns_ranked_candidates(self):
        sample_rate = 8000
        reference = make_burst(sample_rate, 0.02, 500, 2500, 0.5)
        known = RoomCandidate(5.0, 6.0, 2.5, (2.5, 3.0), (2.62, 3.0), 0.6)
        response, backend = simulate_room_response(reference, sample_rate, known, 0.25, backend="analytic")
        result = fit_room_candidates(response, sample_rate, reference, max_candidates=3, backend="analytic")
        self.assertTrue(result.candidates)
        self.assertLessEqual(result.candidates[0].score, result.candidates[-1].score)
        self.assertEqual(result.backend, "analytic")

    def test_zero_input_cannot_produce_room_estimate(self):
        sample_rate = 16000
        reference = make_burst(sample_rate, 0.04, 500, 3500, 0.5)
        response = matched_filter(np.zeros(9000), reference, sample_rate)
        result = fit_room_candidates(response.correlation, sample_rate, reference, backend="analytic", reflection_valid=response.reflection_valid)
        self.assertTrue(response.unreliable)
        self.assertFalse(result.supported)
        self.assertFalse(result.candidates)

    def test_unrelated_noise_cannot_produce_room_estimate(self):
        sample_rate = 16000
        reference = make_burst(sample_rate, 0.04, 500, 3500, 0.5)
        noise = np.random.default_rng(11).normal(0, 0.01, 9000)
        response = matched_filter(noise, reference, sample_rate)
        result = fit_room_candidates(response.correlation, sample_rate, reference, backend="analytic", reflection_valid=response.reflection_valid)
        self.assertFalse(response.reflection_valid)
        self.assertFalse(result.supported)

    def test_direct_path_only_is_not_a_room_reflection(self):
        sample_rate = 16000
        reference = make_burst(sample_rate, 0.04, 500, 3500, 0.5)
        recording = synthetic_recording(reference, sample_rate, delays_s=(0.01,), gains=(1.0,), noise=0)
        response = matched_filter(recording, reference, sample_rate)
        result = fit_room_candidates(response.correlation, sample_rate, reference, backend="analytic", reflection_valid=response.reflection_valid)
        self.assertFalse(response.reflection_valid)
        self.assertFalse(result.supported)

    def test_playback_disabled_is_explicitly_invalid(self):
        sample_rate = 16000
        reference = make_burst(sample_rate, 0.04, 500, 3500, 0.5)
        response = matched_filter(np.zeros(9000), reference, sample_rate)
        self.assertTrue(response.unreliable)
        self.assertIn("No reliable transmitted", response.reflection_reason)

    def test_diagnostic_session_preserves_raw_levels(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = DiagnosticRecorder(16000, np.array([0.0, 0.5, -0.25], dtype=np.float32), directory)
            recorder.add_audio_block(np.array([0.1, -0.2], dtype=np.float32), {"timestamp": 1.0, "output_block": np.array([0.5, 0.0], dtype=np.float32), "status": ""})
            path = recorder.save()
            values = np.load(path)
            self.assertTrue(np.allclose(values["raw_microphone"], [0.1, -0.2]))
            self.assertTrue(np.allclose(values["played_output"], [0.5, 0.0]))

    def test_colocated_shell_has_half_path_radius(self):
        tracker = ReflectionTracker(sound_speed=340.0, microphone_offset=(0, 0, 0), colocated_approximation=True, minimum_repeats=1)
        tracker.update([ReflectionCandidate(0.02, 0.001, 6.8, 0.34, 1.0)])
        shell = tracker.shells()[0]
        x, y, z = shell_surface(shell, (0, 0, 0), ((-10, 10), (-10, 10), (-10, 10)), resolution=12)
        radius = np.sqrt(x[~np.isnan(x)] ** 2 + y[~np.isnan(y)] ** 2 + z[~np.isnan(z)] ** 2)
        self.assertAlmostEqual(float(np.mean(radius)), 3.4, places=2)

    def test_known_relative_echo_delay_sets_shell_path_length(self):
        sample_rate = 16000
        reference = make_burst(sample_rate, 0.04, 500, 3500, 0.5)
        response = matched_filter(synthetic_recording(reference, sample_rate, delays_s=(0.01, 0.035), gains=(1.0, 0.3), noise=0), reference, sample_rate)
        candidates = extract_reflection_candidates(response, sample_rate, 3000, sound_speed=340.0, microphone_offset=(0.12, 0, 0))
        self.assertTrue(candidates)
        self.assertAlmostEqual(candidates[0].relative_delay_s, 0.025, delta=2 / sample_rate)
        self.assertAlmostEqual(candidates[0].path_length_m, 0.12 + 340.0 * 0.025, delta=0.05)

    def test_direct_ringing_guard_rejects_early_correlations(self):
        sample_rate = 16000
        reference = make_burst(sample_rate, 0.04, 500, 3500, 0.5)
        response = matched_filter(synthetic_recording(reference, sample_rate, delays_s=(0.01, 0.035), gains=(1.0, 0.3), noise=0), reference, sample_rate)
        candidates = extract_reflection_candidates(response, sample_rate, 3000, direct_ringing_guard_s=0.04)
        self.assertFalse(candidates)

    def test_view_volume_rejects_impossible_path(self):
        sample_rate = 16000
        reference = make_burst(sample_rate, 0.04, 500, 3500, 0.5)
        response = matched_filter(synthetic_recording(reference, sample_rate, delays_s=(0.01, 0.20), gains=(1.0, 0.3), noise=0), reference, sample_rate)
        candidates = extract_reflection_candidates(response, sample_rate, 3000, max_path_length_m=10.0)
        self.assertTrue(all(candidate.path_length_m <= 10.0 for candidate in candidates))

    def test_noncolocated_shell_uses_direct_separation(self):
        tracker = ReflectionTracker(sound_speed=340.0, microphone_offset=(0.3, 0.4, 0.0), colocated_approximation=False, minimum_repeats=1)
        tracker.update([ReflectionCandidate(0.02, 0.001, 7.1, 0.34, 1.0)])
        shell = tracker.shells()[0]
        x, y, z = shell_surface(shell, (0.3, 0.4, 0.0), ((-10, 10), (-10, 10), (-10, 10)), resolution=16)
        valid = ~np.isnan(x)
        speaker_paths = np.sqrt(x[valid] ** 2 + y[valid] ** 2 + z[valid] ** 2)
        microphone_paths = np.sqrt((x[valid] - 0.3) ** 2 + (y[valid] - 0.4) ** 2 + z[valid] ** 2)
        self.assertTrue(np.allclose(speaker_paths + microphone_paths, 7.1, atol=0.05))

    def test_repeated_measurements_do_not_create_direction(self):
        tracker = ReflectionTracker(minimum_repeats=2, colocated_approximation=True)
        observation = ReflectionCandidate(0.02, 0.001, 6.8, 0.34, 1.0)
        tracker.update([observation])
        tracker.update([observation])
        shells = tracker.shells()
        self.assertEqual(len(shells), 1)
        self.assertTrue(shells[0].colocated_approximation)
        self.assertEqual(shells[0].repeat_count, 2)

    def test_candidate_extraction_caps_sidelobes(self):
        sample_rate = 16000
        reference = make_burst(sample_rate, 0.04, 500, 3500, 0.5)
        recording = synthetic_recording(reference, sample_rate, delays_s=(0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13), gains=(1.0, 0.3, 0.25, 0.2, 0.18, 0.16, 0.14), noise=0)
        response = matched_filter(recording, reference, sample_rate)
        candidates = extract_reflection_candidates(response, sample_rate, 3000.0)
        self.assertLessEqual(len(candidates), 6)

    def test_tracker_retains_shell_through_missed_probes(self):
        candidate = ReflectionCandidate(0.02, 0.001, 6.8, 0.34, 1.0)
        tracker = ReflectionTracker(minimum_repeats=2, max_missed=1)
        tracker.update([candidate])
        tracker.update([candidate])
        self.assertEqual(len(tracker.shells()), 1)
        tracker.update([])
        self.assertEqual(len(tracker.shells()), 1)
        tracker.update([])
        self.assertEqual(len(tracker.shells()), 1)

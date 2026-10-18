import numpy as np
import unittest

from echomap.features import extract_features
from echomap.geometry import path_lengths_from_delays
from echomap.processing import baseline_difference, matched_filter
from echomap.signals import make_burst
from echomap.simulation import synthetic_recording


class PipelineTests(unittest.TestCase):
    def test_matched_filter_finds_known_echo_delay(self):
        sample_rate = 16000
        reference = make_burst(sample_rate, 0.04, 500, 3500, 0.5)
        recording = synthetic_recording(reference, sample_rate, delays_s=(0.01, 0.035), gains=(1.0, 0.3), noise=0)
        response = matched_filter(recording, reference, sample_rate)
        self.assertLess(abs(response.aligned_delay_s - 0.01), 2 / sample_rate)


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

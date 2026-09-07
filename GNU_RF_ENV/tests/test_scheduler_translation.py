"""Phase 3T tests — GNU RF → scheduler translation layer.

Covers all 14 minimum tests (A-N) plus end-to-end GNU RF and equivalence tests.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_GNU_RF_TESTS = str(Path(__file__).resolve().parent)
if _GNU_RF_TESTS not in sys.path:
    sys.path.insert(0, _GNU_RF_TESTS)

_GNU_RF_SCRIPTS = str(Path(__file__).resolve().parent.parent / "scripts")
if _GNU_RF_SCRIPTS not in sys.path:
    sys.path.insert(0, _GNU_RF_SCRIPTS)

_REPO = Path(__file__).resolve().parents[2]
_MASTER_PKG = str(_REPO / "cognitive_ew_smart_scan")
if _MASTER_PKG not in sys.path:
    sys.path.insert(0, _MASTER_PKG)

from scheduler_translation import GnuRfSchedulerTranslation, _band_index  # noqa: E402
from src.receiver.models import ReceiverObservation, DetectionObservation  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_observation(
    center_mhz: float,
    time_us: float = 0.0,
    ibw_mhz: float = 500.0,
    dwell_us: float = 500.0,
    detections=None,
) -> ReceiverObservation:
    """Create a ReceiverObservation with optional detections."""
    return ReceiverObservation(
        time_us=time_us,
        center_frequency_mhz=center_mhz,
        ibw_mhz=ibw_mhz,
        dwell_time_us=dwell_us,
        detections=detections or [],
    )


def _make_detection(
    freq_mhz: float,
    time_us: float = 100.0,
    pw_us: float = 10.0,
    amp_db: float = -100.0,
    aoa_deg: float = 0.0,
    emitter_id: int | None = None,
) -> DetectionObservation:
    return DetectionObservation(
        time_us=time_us,
        frequency_mhz=freq_mhz,
        pulse_width_us=pw_us,
        amplitude_db=amp_db,
        aoa_deg=aoa_deg,
        emitter_id=emitter_id,
    )


# ===================================================================
# A-N minimum tests
# ===================================================================
class TestA_SingleObservationConverts(unittest.TestCase):
    """A. Single ReceiverObservation converts successfully."""

    def test_single_observation_produces_obs(self):
        tr = GnuRfSchedulerTranslation()
        obs = _make_observation(3250.0, detections=[
            _make_detection(3250.0),
        ])
        result = tr.update(obs)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, np.ndarray)


class TestB_OutputShape(unittest.TestCase):
    """B. Output shape matches actual scheduler contract."""

    def test_shape_360(self):
        tr = GnuRfSchedulerTranslation()
        obs = _make_observation(3250.0)
        result = tr.update(obs)
        self.assertEqual(result.shape, (360,))


class TestC_OutputDtype(unittest.TestCase):
    """C. Output dtype matches."""

    def test_dtype_float32(self):
        tr = GnuRfSchedulerTranslation()
        obs = _make_observation(3250.0)
        result = tr.update(obs)
        self.assertEqual(result.dtype, np.float32)


class TestD_OutputBounds(unittest.TestCase):
    """D. Output bounds match."""

    def test_bounds_01(self):
        tr = GnuRfSchedulerTranslation()
        obs = _make_observation(3250.0, detections=[
            _make_detection(3250.0),
        ])
        result = tr.update(obs)
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 1.0))

    def test_bounds_after_multiple_steps(self):
        tr = GnuRfSchedulerTranslation()
        for i in range(10):
            obs = _make_observation(3250.0, time_us=i * 500.0, detections=[
                _make_detection(3250.0, time_us=i * 500.0 + 100.0),
            ])
            result = tr.update(obs)
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 1.0))


class TestE_FeatureOrdering(unittest.TestCase):
    """E. Feature ordering matches the canonical 10-feature layout."""

    def test_feature_layout(self):
        tr = GnuRfSchedulerTranslation()
        det1 = _make_detection(3250.0, time_us=100.0)
        det2 = _make_detection(3250.0, time_us=200.0)
        obs = _make_observation(3250.0, detections=[det1, det2])
        result = tr.update(obs)

        band_start = 6 * 10  # band 6
        occ = result[band_start + 0]    # occupancy
        det_rate = result[band_start + 1]  # detection rate
        miss_rate = result[band_start + 2]  # miss rate
        unc = result[band_start + 3]    # uncertainty
        age = result[band_start + 4]    # revisit age
        emit = result[band_start + 5]   # emitter count
        deint = result[band_start + 6]  # deinterleaver confidence
        per = result[band_start + 7]    # periodicity stability
        agil = result[band_start + 8]   # agility
        prio = result[band_start + 9]   # priority

        # occupancy > 0 (we had a hit)
        self.assertGreater(occ, 0.0)
        # det_rate > 0 (we had detections)
        self.assertGreater(det_rate, 0.0)
        # miss_rate = 1 - det_rate
        self.assertAlmostEqual(float(miss_rate), 1.0 - float(det_rate), places=5)
        # uncertainty in [0, 1]
        self.assertGreaterEqual(unc, 0.0)
        self.assertLessEqual(unc, 1.0)
        # revisit age = 0 (just visited)
        self.assertAlmostEqual(float(age), 0.0, places=5)
        # emitter count >= 0
        self.assertGreaterEqual(emit, 0.0)


class TestF_BandMapping(unittest.TestCase):
    """F. Correct RF band maps to correct scheduler band."""

    def test_3200_mhz_to_band_6(self):
        idx = _band_index(3200.0, 0.0, 18000.0, 36)
        self.assertEqual(idx, 6)

    def test_7999_mhz_to_band_15(self):
        idx = _band_index(7999.75, 0.0, 18000.0, 36)
        self.assertEqual(idx, 15)

    def test_8000_mhz_to_band_16(self):
        idx = _band_index(8000.0, 0.0, 18000.0, 36)
        self.assertEqual(idx, 16)

    def test_center_of_each_band_maps_correctly(self):
        band_width = 18000.0 / 36
        for b in range(36):
            center = b * band_width + band_width / 2.0
            idx = _band_index(center, 0.0, 18000.0, 36)
            self.assertEqual(idx, b, f"Band center {center} MHz mapped to {idx}, expected {b}")


class TestG_SequentialObservations(unittest.TestCase):
    """G. Two sequential observations update state correctly."""

    def test_two_steps_update_state(self):
        tr = GnuRfSchedulerTranslation()

        # Step 1: visit band 6
        obs1 = _make_observation(3250.0, time_us=0.0, detections=[
            _make_detection(3250.0, time_us=50.0),
        ])
        result1 = tr.update(obs1)

        # Band 6 should have occupancy > 0
        self.assertGreater(result1[60], 0.0)  # band 6, feature 0

        # Step 2: visit band 15
        obs2 = _make_observation(7750.0, time_us=500.0, detections=[
            _make_detection(7750.0, time_us=550.0),
        ])
        result2 = tr.update(obs2)

        # Band 15 should now have occupancy > 0
        self.assertGreater(result2[150], 0.0)  # band 15, feature 0

        # Band 6 revisit age should have increased
        band6_age = result2[6 * 10 + 4]  # band 6, feature 4
        self.assertGreater(float(band6_age), 0.0)


class TestH_EmptyObservation(unittest.TestCase):
    """H. Empty/no-detection observation produces valid scheduler input."""

    def test_empty_detections(self):
        tr = GnuRfSchedulerTranslation()
        obs = _make_observation(3250.0, detections=[])
        result = tr.update(obs)
        self.assertEqual(result.shape, (360,))
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 1.0))
        self.assertTrue(np.all(np.isfinite(result)))


class TestI_MultipleBands(unittest.TestCase):
    """I. Multiple bands can be represented."""

    def test_three_bands_accumulated(self):
        tr = GnuRfSchedulerTranslation()

        for center in [250.0, 3250.0, 7750.0]:
            obs = _make_observation(center, detections=[
                _make_detection(center),
            ])
            tr.update(obs)

        result = tr.get_observation()
        # Bands 0, 6, 15 should all have occupancy > 0
        self.assertGreater(result[0 * 10 + 0], 0.0)   # band 0
        self.assertGreater(result[6 * 10 + 0], 0.0)   # band 6
        self.assertGreater(result[15 * 10 + 0], 0.0)  # band 15
        # Other bands should still have occupancy == 0
        self.assertEqual(result[3 * 10 + 0], 0.0)      # band 3


class TestJ_NoGroundTruthRequired(unittest.TestCase):
    """J. No ground-truth field is required."""

    def test_no_emitter_id_needed(self):
        tr = GnuRfSchedulerTranslation()
        det = _make_detection(3250.0, time_us=100.0)
        # Ensure emitter_id is None (no ground truth)
        self.assertIsNone(det.emitter_id)
        obs = _make_observation(3250.0, detections=[det])
        result = tr.update(obs)
        self.assertEqual(result.shape, (360,))
        self.assertTrue(np.all(np.isfinite(result)))


class TestK_Deterministic(unittest.TestCase):
    """K. Same observable input produces deterministic output."""

    def test_same_input_same_output(self):
        tr1 = GnuRfSchedulerTranslation()
        tr2 = GnuRfSchedulerTranslation()

        obs = _make_observation(3250.0, detections=[
            _make_detection(3250.0, time_us=100.0),
        ])
        r1 = tr1.update(obs)
        r2 = tr2.update(obs)
        np.testing.assert_array_equal(r1, r2)

    def test_repeated_input_same_output(self):
        tr = GnuRfSchedulerTranslation()
        obs = _make_observation(3250.0, detections=[
            _make_detection(3250.0, time_us=100.0),
        ])
        tr.update(obs)
        tr.update(obs)
        r1 = tr.get_observation()
        r2 = tr.get_observation()
        np.testing.assert_array_equal(r1, r2)


class TestL_EmitterIdIrrelevant(unittest.TestCase):
    """L. Renaming/permuting emitter IDs does not alter scheduler output."""

    def test_different_emitter_ids_same_output(self):
        tr1 = GnuRfSchedulerTranslation()
        tr2 = GnuRfSchedulerTranslation()

        det1 = _make_detection(3250.0, time_us=100.0, emitter_id=42)
        det2 = _make_detection(3250.0, time_us=100.0, emitter_id=99)

        obs1 = _make_observation(3250.0, detections=[det1])
        obs2 = _make_observation(3250.0, detections=[det2])

        r1 = tr1.update(obs1)
        r2 = tr2.update(obs2)
        np.testing.assert_array_equal(r1, r2)


class TestM_RawIQNotAccepted(unittest.TestCase):
    """M. Raw IQ is not accepted as scheduler input."""

    def test_raw_array_produces_uninformed_observation(self):
        tr = GnuRfSchedulerTranslation()
        raw_iq = np.random.randn(1000).astype(np.complex128)
        result = tr.update(raw_iq)
        self.assertEqual(result.shape, (360,))
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 1.0))
        self.assertEqual(float(result[6 * 10 + 0]), 0.0)


class TestN_NoActionGeneration(unittest.TestCase):
    """N. Scheduler action is not generated by this component."""

    def test_no_select_action(self):
        tr = GnuRfSchedulerTranslation()
        self.assertFalse(hasattr(tr, "select_action"))
        self.assertFalse(hasattr(tr, "choose_band"))
        self.assertFalse(hasattr(tr, "act"))

    def test_update_returns_obs_not_action(self):
        tr = GnuRfSchedulerTranslation()
        obs = _make_observation(3250.0)
        result = tr.update(obs)
        self.assertEqual(result.shape, (360,))
        self.assertEqual(result.dtype, np.float32)


# ===================================================================
# Step 16 — End-to-end GNU RF test
# ===================================================================
class TestEndToEndGnuRf(unittest.TestCase):
    """Step 16: PerTuneGenerator → ... → scheduler observation."""

    def test_3200_mhz_end_to_end(self):
        from per_tune_generator import PerTuneGenerator, EmitterConfig
        from iq_to_pdw import PDWDetector
        from frequency_context import FrequencyContext
        from iq_bridge import IQReceiverBridge

        center = 3250.0
        emitters = [EmitterConfig(rf_frequency_mhz=3250.1, pri_us=100.0, pulse_width_us=10.0)]

        gen = PerTuneGenerator()
        gen.configure(center_frequency_mhz=center, sample_rate=2e6, noise_amplitude=0.02, noise_seed=42)
        for e in emitters:
            gen.add_emitter(e)
        iq = gen.generate_iq(duration_us=500.0, include_noise=True)

        detector = PDWDetector(sample_rate=2e6, threshold_db_above_noise=5.0)
        pdws = detector.detect_iq(iq, sample_offset=0)

        ctx = FrequencyContext(center_frequency_mhz=center)
        bridge = IQReceiverBridge(ctx)
        detections = []
        for pdw in pdws:
            pulse = bridge.pdw_to_pulse(pdw)
            detections.append(_make_detection(
                freq_mhz=pulse["frequency_mhz"],
                time_us=pulse["toa_us"],
                pw_us=pulse["pulse_width_us"],
                amp_db=pulse["amplitude_db"],
                aoa_deg=pulse["aoa_deg"],
            ))

        tr = GnuRfSchedulerTranslation()
        obs = _make_observation(center, detections=detections)
        result = tr.update(obs)

        self.assertEqual(result.shape, (360,))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 1.0))

        # Band 6 (3000-3500 MHz) should show activity
        band6_occ = result[6 * 10 + 0]
        self.assertGreater(band6_occ, 0.0, "Band 6 should show occupancy")

    def test_8000_mhz_end_to_end(self):
        from per_tune_generator import PerTuneGenerator, EmitterConfig
        from iq_to_pdw import PDWDetector
        from frequency_context import FrequencyContext
        from iq_bridge import IQReceiverBridge

        center = 7750.0
        emitters = [EmitterConfig(rf_frequency_mhz=7999.75, pri_us=200.0, pulse_width_us=10.0)]

        gen = PerTuneGenerator()
        gen.configure(center_frequency_mhz=center, sample_rate=2e6, noise_amplitude=0.02, noise_seed=42)
        for e in emitters:
            gen.add_emitter(e)
        iq = gen.generate_iq(duration_us=500.0, include_noise=True)

        detector = PDWDetector(sample_rate=2e6, threshold_db_above_noise=5.0)
        pdws = detector.detect_iq(iq, sample_offset=0)

        ctx = FrequencyContext(center_frequency_mhz=center)
        bridge = IQReceiverBridge(ctx)
        detections = []
        for pdw in pdws:
            pulse = bridge.pdw_to_pulse(pdw)
            detections.append(_make_detection(
                freq_mhz=pulse["frequency_mhz"],
                time_us=pulse["toa_us"],
                pw_us=pulse["pulse_width_us"],
                amp_db=pulse["amplitude_db"],
                aoa_deg=pulse["aoa_deg"],
            ))

        tr = GnuRfSchedulerTranslation()
        obs = _make_observation(center, detections=detections)
        result = tr.update(obs)

        self.assertEqual(result.shape, (360,))
        self.assertTrue(np.all(np.isfinite(result)))

        # Band 15 (7500-8000 MHz) should show activity
        band15_occ = result[15 * 10 + 0]
        self.assertGreater(band15_occ, 0.0, "Band 15 should show occupancy")


# ===================================================================
# Step 17 — Equivalence test
# ===================================================================
class TestEquivalence(unittest.TestCase):
    """Step 17: Structured env and GNU RF produce equivalent observation properties."""

    def test_equivalent_properties(self):
        """Both paths produce same shape/dtype/bounds/feature_ordering."""
        from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv

        config = {
            "n_bands": 36,
            "freq_min_mhz": 0.0,
            "freq_max_mhz": 18000.0,
            "ibw_mhz": 500.0,
            "dwell_time_us": 500.0,
            "frequency_step_mhz": 500.0,
            "detection_threshold_db": -140.0,
            "max_steps_per_episode": 10,
        }

        # Path A: existing env (no records, just reset + step)
        env = CognitiveRFScanEnv(config=config)
        env.reset(seed=42)
        obs_a, _, _, _, _ = env.step(6)  # visit band 6

        # Path B: translation layer
        tr = GnuRfSchedulerTranslation()
        det = _make_detection(3250.0, time_us=100.0)
        obs_b = tr.update(_make_observation(3250.0, detections=[det]))

        # Same shape
        self.assertEqual(obs_a.shape, obs_b.shape)
        self.assertEqual(obs_a.shape, (360,))

        # Same dtype
        self.assertEqual(obs_a.dtype, obs_b.dtype)
        self.assertEqual(obs_a.dtype, np.float32)

        # Same bounds
        self.assertTrue(np.all(obs_a >= 0.0) and np.all(obs_a <= 1.0))
        self.assertTrue(np.all(obs_b >= 0.0) and np.all(obs_b <= 1.0))

        # Same feature ordering: translation path shows occupancy for a hit band
        self.assertGreater(obs_b[6 * 10 + 0], 0.0)  # translation band 6 occupancy

        # Env with records=[] has no pulses, so band 6 occupancy is 0 (expected)
        self.assertEqual(float(obs_a[6 * 10 + 0]), 0.0)  # env band 6: no records = no hit

        # Both produce finite values
        self.assertTrue(np.all(np.isfinite(obs_a)))
        self.assertTrue(np.all(np.isfinite(obs_b)))

        # Same band semantics: band 6 is [3000, 3500) MHz
        from scheduler_translation import _band_index
        self.assertEqual(_band_index(3250.0, 0.0, 18000.0, 36), 6)


if __name__ == "__main__":
    unittest.main()

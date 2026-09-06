"""Unit tests for per_tune_generator.py — Phase 3O.

Tests the PerTuneConfig, EmitterConfig, and PerTuneGenerator classes.
No GNU Radio dependency required for these tests.

Test matrix:
  A. Baseband formula — center=3200, RF=3200.1 → +100 kHz
  B. Baseband formula — center=8000, RF=7999.75 → -250 kHz
  C. Baseband formula — same center and RF → 0 kHz
  D. PerTuneConfig validate — positive center passes
  E. PerTuneConfig validate — non-finite center raises
  F. PerTuneConfig validate — negative center raises
  G. PerTuneConfig validate — zero sample_rate raises
  H. EmitterConfig — zero PRI > PW passes
  I. EmitterConfig — PRI <= PW raises
  J. EmitterConfig — negative amplitude raises
  K. EmitterConfig — negative jitter raises
  L. Generator — configure + no emitters → zeros
  M. Generator — single emitter → non-zero IQ
  N. Generator — deterministic (same seed → same output)
  O. Generator — include_noise=False → no noise
  P. Generator — multi-emitter superposition
  Q. Generator — add_emitter before configure raises
  R. Generator — invalid emitter type raises
  S. Generator — output dtype is complex64
  T. End-to-end — generator → PDWDetector → FrequencyContext → IQReceiverBridge
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

# Ensure scripts/ is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
# Ensure the authoritative master receiver is importable (read-only use).
_MASTER_SRC = str(Path(__file__).resolve().parents[2] / "cognitive_ew_smart_scan" / "src")
if _MASTER_SRC not in sys.path:
    sys.path.insert(0, _MASTER_SRC)

from per_tune_generator import EmitterConfig, PerTuneConfig, PerTuneGenerator


# ================================================================
# A-C: EmitterConfig.baseband_frequency_khz
# ================================================================


class TestEmitterBasebandFormula(unittest.TestCase):
    """Verify local_frequency_khz = (rf_frequency_mhz - center_frequency_mhz) * 1000."""

    def test_a_positive_offset(self):
        """A. center=3200, RF=3200.1 → +100 kHz."""
        em = EmitterConfig(rf_frequency_mhz=3200.1)
        result = em.baseband_frequency_khz(center_frequency_mhz=3200.0)
        self.assertAlmostEqual(result, 100.0)

    def test_b_negative_offset(self):
        """B. center=8000, RF=7999.75 → -250 kHz."""
        em = EmitterConfig(rf_frequency_mhz=7999.75)
        result = em.baseband_frequency_khz(center_frequency_mhz=8000.0)
        self.assertAlmostEqual(result, -250.0)

    def test_c_zero_offset(self):
        """C. Same center and RF → 0 kHz."""
        em = EmitterConfig(rf_frequency_mhz=3200.0)
        result = em.baseband_frequency_khz(center_frequency_mhz=3200.0)
        self.assertAlmostEqual(result, 0.0)


# ================================================================
# D-H: PerTuneConfig validation
# ================================================================


class TestPerTuneConfigValidation(unittest.TestCase):
    """PerTuneConfig.validate() edge cases."""

    def test_d_positive_center_passes(self):
        """D. Valid config passes validation."""
        cfg = PerTuneConfig(center_frequency_mhz=3200.0)
        cfg.validate()

    def test_e_non_finite_center_raises(self):
        """E. NaN center raises ValueError."""
        cfg = PerTuneConfig(center_frequency_mhz=float("nan"))
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_e_inf_center_raises(self):
        """E+. Inf center raises ValueError."""
        cfg = PerTuneConfig(center_frequency_mhz=float("inf"))
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_f_negative_center_raises(self):
        """F. Negative center raises ValueError."""
        cfg = PerTuneConfig(center_frequency_mhz=-100.0)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_g_zero_sample_rate_raises(self):
        """G. Zero sample_rate raises ValueError."""
        cfg = PerTuneConfig(center_frequency_mhz=3200.0, sample_rate=0)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_g_negative_noise_raises(self):
        """G+. Negative noise_amplitude raises ValueError."""
        cfg = PerTuneConfig(center_frequency_mhz=3200.0, noise_amplitude=-0.1)
        with self.assertRaises(ValueError):
            cfg.validate()


# ================================================================
# I-K: EmitterConfig validation (via add_emitter)
# ================================================================


class TestEmitterValidation(unittest.TestCase):
    """EmitterConfig constraints enforced by PerTuneGenerator.add_emitter."""

    def setUp(self):
        self.gen = PerTuneGenerator()
        self.gen.configure(center_frequency_mhz=3200.0)

    def test_i_pri_gt_pw_passes(self):
        """I. PRI > PW passes validation."""
        em = EmitterConfig(rf_frequency_mhz=3200.1, pulse_width_us=10, pri_us=100)
        self.gen.add_emitter(em)

    def test_i_pri_equals_pw_raises(self):
        """I+. PRI == PW raises ValueError."""
        em = EmitterConfig(rf_frequency_mhz=3200.1, pulse_width_us=10, pri_us=10)
        with self.assertRaises(ValueError):
            self.gen.add_emitter(em)

    def test_j_pri_lt_pw_raises(self):
        """J. PRI < PW raises ValueError."""
        em = EmitterConfig(rf_frequency_mhz=3200.1, pulse_width_us=10, pri_us=5)
        with self.assertRaises(ValueError):
            self.gen.add_emitter(em)

    def test_k_negative_amplitude_raises(self):
        """K. Negative amplitude raises ValueError."""
        em = EmitterConfig(rf_frequency_mhz=3200.1, amplitude=-1.0)
        with self.assertRaises(ValueError):
            self.gen.add_emitter(em)

    def test_k_negative_jitter_raises(self):
        """K+. Negative jitter_fraction raises ValueError."""
        em = EmitterConfig(rf_frequency_mhz=3200.1, jitter_fraction=-0.01)
        with self.assertRaises(ValueError):
            self.gen.add_emitter(em)

    def test_k_nonfinite_amplitude_raises(self):
        """K++. Non-finite amplitude must be rejected."""
        em = EmitterConfig(rf_frequency_mhz=3200.1, amplitude=float("inf"))
        with self.assertRaises(ValueError):
            self.gen.add_emitter(em)

    def test_k_nonfinite_jitter_raises(self):
        """K+++. Non-finite jitter_fraction must be rejected."""
        em = EmitterConfig(rf_frequency_mhz=3200.1, jitter_fraction=float("nan"))
        with self.assertRaises(ValueError):
            self.gen.add_emitter(em)


# ================================================================
# L-S: PerTuneGenerator IQ generation
# ================================================================


class TestPerTuneGenerator(unittest.TestCase):
    """Generator behavior: deterministic IQ, noise, multi-emitter, dtype."""

    def test_l_no_emitters_zeros(self):
        """L. No emitters → all-zero IQ (noise disabled)."""
        gen = PerTuneGenerator()
        gen.configure(center_frequency_mhz=3200.0, noise_amplitude=0.0)
        iq = gen.generate_iq(duration_us=100.0, include_noise=False)
        self.assertEqual(iq.shape, (200,))
        np.testing.assert_array_equal(iq, 0.0)

    def test_m_single_emitter_nonzero(self):
        """M. One emitter → non-zero IQ."""
        gen = PerTuneGenerator()
        gen.configure(center_frequency_mhz=3200.0, noise_amplitude=0.0)
        gen.add_emitter(EmitterConfig(rf_frequency_mhz=3200.1, amplitude=1.0))
        iq = gen.generate_iq(duration_us=100.0, include_noise=False)
        self.assertEqual(iq.shape, (200,))
        self.assertGreater(np.max(np.abs(iq)), 0.0)

    def test_n_deterministic(self):
        """N. Same seed → identical output."""
        gen1 = PerTuneGenerator()
        gen1.configure(center_frequency_mhz=3200.0, noise_amplitude=0.05, noise_seed=42)
        gen1.add_emitter(EmitterConfig(rf_frequency_mhz=3200.1, seed=99))
        iq1 = gen1.generate_iq(duration_us=200.0)

        gen2 = PerTuneGenerator()
        gen2.configure(center_frequency_mhz=3200.0, noise_amplitude=0.05, noise_seed=42)
        gen2.add_emitter(EmitterConfig(rf_frequency_mhz=3200.1, seed=99))
        iq2 = gen2.generate_iq(duration_us=200.0)

        np.testing.assert_array_equal(iq1, iq2)

    def test_o_no_noise(self):
        """O. include_noise=False → no noise or channel applied."""
        gen = PerTuneGenerator()
        gen.configure(center_frequency_mhz=3200.0, noise_amplitude=10.0, noise_seed=7)
        gen.add_emitter(EmitterConfig(rf_frequency_mhz=3200.1, amplitude=1.0))

        iq_noisy = gen.generate_iq(duration_us=50.0, include_noise=True)
        iq_clean = gen.generate_iq(duration_us=50.0, include_noise=False)

        # Clean signal should be deterministic and noise-free;
        # noisy version should differ from clean.
        self.assertFalse(np.array_equal(iq_noisy, iq_clean))

    def test_p_multi_emitter_superposition(self):
        """P. Two emitters → larger combined amplitude than single."""
        gen_single = PerTuneGenerator()
        gen_single.configure(center_frequency_mhz=3200.0, noise_amplitude=0.0)
        gen_single.add_emitter(EmitterConfig(rf_frequency_mhz=3200.1, amplitude=1.0))
        iq_single = gen_single.generate_iq(duration_us=100.0, include_noise=False)

        gen_dual = PerTuneGenerator()
        gen_dual.configure(center_frequency_mhz=3200.0, noise_amplitude=0.0)
        gen_dual.add_emitter(EmitterConfig(rf_frequency_mhz=3200.1, amplitude=1.0))
        gen_dual.add_emitter(EmitterConfig(rf_frequency_mhz=3200.0, amplitude=1.0))
        iq_dual = gen_dual.generate_iq(duration_us=100.0, include_noise=False)

        # Dual should have at least as much energy as single.
        self.assertGreaterEqual(
            float(np.max(np.abs(iq_dual))),
            float(np.max(np.abs(iq_single))) - 1e-6,
        )

    def test_q_add_before_configure_raises(self):
        """Q. add_emitter before configure → RuntimeError."""
        gen = PerTuneGenerator()
        with self.assertRaises(RuntimeError):
            gen.add_emitter(EmitterConfig(rf_frequency_mhz=3200.1))

    def test_r_wrong_type_raises(self):
        """R. Non-EmitterConfig → TypeError."""
        gen = PerTuneGenerator()
        gen.configure(center_frequency_mhz=3200.0)
        with self.assertRaises(TypeError):
            gen.add_emitter("not an emitter")  # type: ignore

    def test_s_dtype_complex64(self):
        """S. Output dtype is complex64."""
        gen = PerTuneGenerator()
        gen.configure(center_frequency_mhz=3200.0, noise_amplitude=0.05)
        gen.add_emitter(EmitterConfig(rf_frequency_mhz=3200.1))
        iq = gen.generate_iq(duration_us=100.0)
        self.assertEqual(iq.dtype, np.complex64)

    def test_properties_before_configure_raise(self):
        """Properties raise RuntimeError if configure() not called."""
        gen = PerTuneGenerator()
        with self.assertRaises(RuntimeError):
            _ = gen.center_frequency_mhz
        with self.assertRaises(RuntimeError):
            _ = gen.sample_rate


# ================================================================
# T: End-to-end integration
# ================================================================


class TestEndToEnd(unittest.TestCase):
    """Generator → PDWDetector → FrequencyContext → IQReceiverBridge → SieveReceiver."""

    def test_t_full_pipeline(self):
        """T. Generate IQ → detect PDWs → map frequency → bridge → SieveReceiver."""
        from iq_to_pdw import PDWDetector
        from frequency_context import FrequencyContext
        from iq_bridge import IQReceiverBridge
        from receiver import SieveReceiver

        center_mhz = 3200.0
        rf_mhz = 3200.1

        # 1. Generate IQ with the per-tune generator.
        gen = PerTuneGenerator()
        gen.configure(center_frequency_mhz=center_mhz, noise_amplitude=0.02, noise_seed=7)
        gen.add_emitter(EmitterConfig(
            rf_frequency_mhz=rf_mhz,
            pulse_width_us=20.0,
            pri_us=200.0,
            amplitude=1.0,
            seed=42,
        ))
        iq = gen.generate_iq(duration_us=1000.0, include_noise=True)
        self.assertEqual(iq.dtype, np.complex64)
        self.assertGreater(len(iq), 0)

        # 2. Detect PDWs.
        detector = PDWDetector(sample_rate=2e6, threshold_db_above_noise=5.0)
        pdws = detector.detect_iq(iq, sample_offset=0)
        self.assertGreater(len(pdws), 0, "Should detect at least one pulse")

        for pdw in pdws:
            self.assertIn("frequency_local_khz", pdw)
            self.assertIn("pulse_width_us", pdw)
            self.assertIn("toa_us", pdw)
            # Ground-truth isolation: no emitter_id.
            self.assertNotIn("emitter_id", pdw)

        # 3. Map local → RF via FrequencyContext.
        fctx = FrequencyContext(center_frequency_mhz=center_mhz)
        for pdw in pdws:
            rf_est = fctx.local_to_rf(pdw["frequency_local_khz"])
            # Within 1 MHz of true RF (generous; noise + two-tap channel).
            self.assertTrue(
                abs(rf_est - rf_mhz) < 1.0,
                f"RF estimate {rf_est:.3f} too far from true {rf_mhz}",
            )

        # 4. Bridge to SieveReceiver pulse format.
        bridge = IQReceiverBridge(frequency_context=fctx)
        for pdw in pdws:
            pulse = bridge.pdw_to_pulse(pdw)
            self.assertIn("frequency_mhz", pulse)
            self.assertIn("toa_us", pulse)
            self.assertIn("pulse_width_us", pulse)
            self.assertIn("pulse_id", pulse)

        # 5. Feed to SieveReceiver via bridge helper.
        receiver = SieveReceiver()
        observations = bridge.process_pdws_with_detection(receiver, pdws)
        self.assertIsInstance(observations, list)


if __name__ == "__main__":
    unittest.main()

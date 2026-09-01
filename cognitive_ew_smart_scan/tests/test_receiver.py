"""Receiver unit tests ported from the friend repository."""

from __future__ import annotations

import math
import unittest

from src.receiver import SieveReceiver, ReceiverConfigError, to_hz, to_ghz

RX_KW = {
    "total_bandwidth": 18e3,
    "ibw": 1e3,
    "frequency_step": 500.0,
    "dwell_time": 100.0,
}


def pulse(freq_mhz, toa_us, width_us, amp_db=-100.0, aoa_deg=45.0, pulse_id=None):
    return {
        "frequency_mhz": float(freq_mhz),
        "toa_us": float(toa_us),
        "pulse_width_us": float(width_us),
        "exit_us": float(toa_us) + float(width_us),
        "amplitude_db": float(amp_db),
        "aoa_deg": float(aoa_deg),
        "pulse_id": pulse_id,
    }


class TestInitialization(unittest.TestCase):
    def test_center_frequency_valid(self):
        r = SieveReceiver(**RX_KW)
        self.assertEqual(r.center_frequency_mhz, r.legal_center_min_mhz)
        self.assertEqual(r.center_frequency_mhz, 500.0)

    def test_current_time_zero(self):
        r = SieveReceiver(**RX_KW)
        self.assertEqual(r.current_time_us, 0.0)

    def test_ibw_bounds(self):
        r = SieveReceiver(**RX_KW)
        self.assertEqual(r.legal_center_min_mhz, 500.0)
        self.assertEqual(r.legal_center_max_mhz, 17500.0)


class TestConfigValidation(unittest.TestCase):
    def test_negative_bandwidth(self):
        with self.assertRaises(ReceiverConfigError):
            SieveReceiver(total_bandwidth=-1, ibw=1e3, frequency_step=500, dwell_time=100)

    def test_zero_ibw(self):
        with self.assertRaises(ReceiverConfigError):
            SieveReceiver(total_bandwidth=18e3, ibw=0, frequency_step=500, dwell_time=100)

    def test_ibw_exceeds_bandwidth(self):
        with self.assertRaises(ReceiverConfigError):
            SieveReceiver(total_bandwidth=1e3, ibw=2e3, frequency_step=500, dwell_time=100)

    def test_negative_step(self):
        with self.assertRaises(ReceiverConfigError):
            SieveReceiver(total_bandwidth=18e3, ibw=1e3, frequency_step=-500, dwell_time=100)

    def test_negative_dwell(self):
        with self.assertRaises(ReceiverConfigError):
            SieveReceiver(total_bandwidth=18e3, ibw=1e3, frequency_step=500, dwell_time=-1)

    def test_nan_config_rejected(self):
        with self.assertRaises(ReceiverConfigError):
            SieveReceiver(total_bandwidth=float("nan"), ibw=1e3, frequency_step=500, dwell_time=100)


class TestReset(unittest.TestCase):
    def test_reset_deterministic_and_clears_state(self):
        r = SieveReceiver(**RX_KW)
        r.tune(8000)
        r.dwell()
        r.scan_once(pulses=[pulse(8000, toa_us=100, width_us=50)])
        self.assertEqual(len(r.detections), 1)
        r.reset()
        self.assertEqual(r.center_frequency_mhz, 500.0)
        self.assertEqual(r.current_time_us, 0.0)
        self.assertEqual(r.detections, [])
        self.assertIsNone(r.last_observation)


class TestTuning(unittest.TestCase):
    def test_valid_tune(self):
        r = SieveReceiver(**RX_KW)
        self.assertEqual(r.tune(8000), 8000.0)
        self.assertEqual(r.center_frequency_mhz, 8000.0)

    def test_tune_below_lower_clipped(self):
        r = SieveReceiver(**RX_KW)
        r.tune(0)
        self.assertEqual(r.center_frequency_mhz, r.legal_center_min_mhz)

    def test_tune_above_upper_clipped(self):
        r = SieveReceiver(**RX_KW)
        r.tune(1e9)
        self.assertEqual(r.center_frequency_mhz, r.legal_center_max_mhz)

    def test_tune_nan_rejected(self):
        r = SieveReceiver(**RX_KW)
        with self.assertRaises(ValueError):
            r.tune(float("nan"))


class TestFrequencyWindow(unittest.TestCase):
    def test_exact_ibw(self):
        r = SieveReceiver(**RX_KW)
        r.tune(8000)
        lower, upper = r.get_frequency_window()
        self.assertAlmostEqual(upper - lower, 1000.0)
        self.assertEqual(lower, 7500.0)
        self.assertEqual(upper, 8500.0)


class TestStep(unittest.TestCase):
    def test_step_up(self):
        r = SieveReceiver(**RX_KW)
        r.tune(8000)
        self.assertEqual(r.step_up(), 8500.0)
        self.assertEqual(r.step_up(), 9000.0)

    def test_step_down(self):
        r = SieveReceiver(**RX_KW)
        r.tune(8000)
        self.assertEqual(r.step_down(), 7500.0)

    def test_step_up_clips_at_max(self):
        r = SieveReceiver(**RX_KW)
        r.tune(r.legal_center_max_mhz)
        self.assertEqual(r.step_up(), r.legal_center_max_mhz)

    def test_step_down_clips_at_min(self):
        r = SieveReceiver(**RX_KW)
        r.tune(r.legal_center_min_mhz)
        self.assertEqual(r.step_down(), r.legal_center_min_mhz)


class TestDwell(unittest.TestCase):
    def test_time_advances(self):
        r = SieveReceiver(**RX_KW)
        self.assertEqual(r.current_time_us, 0.0)
        r.dwell()
        self.assertEqual(r.current_time_us, 100.0)
        r.dwell()
        self.assertEqual(r.current_time_us, 200.0)


class TestUnitConsistency(unittest.TestCase):
    def test_mhz_to_hz_and_ghz(self):
        self.assertEqual(to_hz(3199.19), 3199190000.0)
        self.assertEqual(to_ghz(3199.19), 3.19919)

    def test_receiver_uses_mhz_not_hz(self):
        r = SieveReceiver(**RX_KW)
        r.tune(8000)
        lower, upper = r.get_frequency_window()
        self.assertTrue(7000 <= lower <= 8000 <= upper <= 9000)


class TestFrequencyVisibility(unittest.TestCase):
    def test_pulse_inside_ibw_candidate(self):
        r = SieveReceiver(**RX_KW)
        r.tune(8000)
        _ = r.dwell()
        det = r.process_pulse(pulse(7990, toa_us=100, width_us=50))
        self.assertIsNotNone(det)
        self.assertTrue(det.detected)

    def test_pulse_outside_ibw_not_visible(self):
        r = SieveReceiver(**RX_KW)
        r.tune(8000)
        _ = r.dwell()
        self.assertIsNone(r.process_pulse(pulse(12000, toa_us=100, width_us=50)))

    def test_pulse_on_window_boundary_visible(self):
        r = SieveReceiver(**RX_KW)
        r.tune(8000)
        _ = r.dwell()
        self.assertIsNotNone(r.process_pulse(pulse(7500, toa_us=100, width_us=50)))
        self.assertIsNotNone(r.process_pulse(pulse(8500, toa_us=100, width_us=50)))


class TestPulseTiming(unittest.TestCase):
    def _rx_at(self, t_us):
        r = SieveReceiver(**RX_KW)
        r.tune(8000)
        while r.current_time_us < t_us:
            r.dwell()
        return r

    def test_pulse_before_dwell_not_detected(self):
        r = self._rx_at(200)
        self.assertIsNone(r.process_pulse(pulse(8000, toa_us=50, width_us=25)))

    def test_pulse_active_during_dwell_detected(self):
        r = self._rx_at(100)
        det = r.process_pulse(pulse(8000, toa_us=100, width_us=200))
        self.assertIsNotNone(det)

    def test_pulse_begins_during_dwell_detected_at_its_toa(self):
        r = self._rx_at(150)
        det = r.process_pulse(pulse(8000, toa_us=150, width_us=100))
        self.assertIsNotNone(det)

    def test_pulse_ends_during_dwell_not_detected_after_end(self):
        r = self._rx_at(150)
        self.assertIsNone(r.process_pulse(pulse(8000, toa_us=100, width_us=40)))

    def test_pulse_ending_exactly_at_sample_not_detected_half_open(self):
        r = self._rx_at(100)
        self.assertIsNone(r.process_pulse(pulse(8000, toa_us=50, width_us=50)))


class TestDetectionThreshold(unittest.TestCase):
    def test_amplitude_below_threshold_not_detected(self):
        r = SieveReceiver(**RX_KW, detection_threshold_db=-90.0)
        r.tune(8000)
        _ = r.dwell()
        self.assertIsNone(r.process_pulse(pulse(8000, toa_us=100, width_us=50, amp_db=-100.0)))

    def test_amplitude_at_threshold_detected(self):
        r = SieveReceiver(**RX_KW, detection_threshold_db=-90.0)
        r.tune(8000)
        _ = r.dwell()
        det = r.process_pulse(pulse(8000, toa_us=100, width_us=50, amp_db=-90.0))
        self.assertIsNotNone(det)

    def test_amplitude_above_threshold_detected(self):
        r = SieveReceiver(**RX_KW, detection_threshold_db=-90.0)
        r.tune(8000)
        _ = r.dwell()
        det = r.process_pulse(pulse(8000, toa_us=100, width_us=50, amp_db=-80.0))
        self.assertIsNotNone(det)


if __name__ == "__main__":
    unittest.main()

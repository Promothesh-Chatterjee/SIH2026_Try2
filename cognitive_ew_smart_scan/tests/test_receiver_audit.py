"""Audit-oriented integration checks for the receiver path."""

import tempfile
import unittest
from pathlib import Path

from src.receiver import SieveReceiver, ReceiverObservation


class TestReceiverAudit(unittest.TestCase):
    def _rx(self, **over):
        kw = {
            "total_bandwidth": 18000.0,
            "ibw": 1000.0,
            "frequency_step": 500.0,
            "dwell_time": 100.0,
            "detection_threshold_db": -140.0,
        }
        kw.update(over)
        return SieveReceiver(**kw)

    def test_dwell_interval_is_recorded(self):
        r = self._rx()
        obs = r.scan_once()
        self.assertEqual(obs.dwell_interval_us, [0.0, 100.0])
        self.assertIn("dwell_interval_us", obs.to_dict())

    def test_pulse_buffer_tracks_active_pulses(self):
        r = self._rx()
        r.add_pulse({"frequency_mhz": 3000.0, "toa_us": 100.0, "exit_us": 200.0, "pulse_width_us": 100.0, "amplitude_db": -120.0, "aoa_deg": 90.0, "pulse_id": 1})
        self.assertEqual(len(r.buffered_pulses()), 1)
        r.remove_pulse(1)
        self.assertEqual(len(r.buffered_pulses()), 0)

    def test_scan_advances_without_retuning_to_pulse(self):
        r = self._rx()
        r.add_pulse({"frequency_mhz": 12000.0, "toa_us": 0.0, "exit_us": 10.0, "pulse_width_us": 10.0, "amplitude_db": -120.0, "aoa_deg": 90.0, "pulse_id": 1})
        center_before = r.center_frequency_mhz
        r.scan_once()
        self.assertEqual(r.center_frequency_mhz, center_before + r.frequency_step_mhz)

    def test_receiver_observation_has_expected_shape(self):
        r = self._rx()
        r.tune(3000.0)
        r.add_pulse({"frequency_mhz": 3000.0, "toa_us": 210.0, "exit_us": 220.0, "pulse_width_us": 10.0, "amplitude_db": -120.0, "aoa_deg": 90.0, "pulse_id": 0})
        r.advance(200.0)
        obs = r.scan_once()
        self.assertIsInstance(obs, ReceiverObservation)
        self.assertEqual(obs.center_frequency_mhz, 3000.0)
        self.assertGreaterEqual(len(obs.detections), 1)

    def test_end_to_end_receiver_chain(self):
        d = Path(tempfile.mkdtemp()) / "controlled.txt"
        d.write_text("record_1: data=[100.0, 3200.0, 30.0, -120.0, 90.0], label=1\n", encoding="utf-8")
        self.assertTrue(d.exists())
        r = self._rx()
        r.tune(3200.0)
        r.advance(100.0)
        r.add_pulse({"frequency_mhz": 3200.0, "toa_us": 100.0, "exit_us": 130.0, "pulse_width_us": 30.0, "amplitude_db": -120.0, "aoa_deg": 90.0, "pulse_id": 1, "emitter_id": 1})
        obs = r.scan_once()
        self.assertEqual(sorted(d.frequency_mhz for d in obs.detections), [3200.0])


if __name__ == "__main__":
    unittest.main()

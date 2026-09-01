"""Integration tests for the RadioEnvironment -> SieveReceiver flow."""

import unittest

from src.environment.radio_environment import ActivePulse, PulseRecord, RadioEnvironment
from src.receiver import SieveReceiver, attach_receiver


def make_record(*, toa_us: float, frequency_mhz: float, pulse_width_us: float, amplitude_db: float = -100.0, aoa_deg: float = 45.0, emitter_id: int = 1, source_id: str = "integration") -> PulseRecord:
    return PulseRecord(
        toa_us=float(toa_us),
        frequency_mhz=float(frequency_mhz),
        pulse_width_us=float(pulse_width_us),
        amplitude_db=float(amplitude_db),
        aoa_deg=float(aoa_deg),
        emitter_id=int(emitter_id),
        source_id=source_id,
    )


class TestReceiverEnvironmentIntegration(unittest.TestCase):
    def test_environment_feeds_receiver_without_manual_retune(self):
        records = [
            make_record(toa_us=10.0, frequency_mhz=750.0, pulse_width_us=50.0, emitter_id=1),
            make_record(toa_us=120.0, frequency_mhz=1400.0, pulse_width_us=50.0, emitter_id=2),
            make_record(toa_us=120.0, frequency_mhz=3000.0, pulse_width_us=50.0, emitter_id=3),
            make_record(toa_us=120.0, frequency_mhz=1300.0, pulse_width_us=50.0, emitter_id=4),
        ]

        env = RadioEnvironment(records)
        receiver = SieveReceiver(total_bandwidth=18e3, ibw=1e3, frequency_step=500.0, dwell_time=100.0, detection_threshold_db=-140.0)

        bridge = attach_receiver(env, receiver)
        self.assertIsNotNone(bridge)
        self.assertEqual(receiver.center_frequency_mhz, 500.0)

        env.run()

        self.assertGreaterEqual(receiver.current_time_us, 120.0)
        self.assertEqual(receiver.center_frequency_mhz, 1000.0)

        detected_ids = {d.pulse_id for d in receiver.detection_history}
        self.assertIn(0, detected_ids)

        detections_at_120 = [d for d in receiver.detection_history if abs(d.time_us - 120.0) < 1e-9]
        frequencies_at_120 = {round(d.frequency_mhz, 6) for d in detections_at_120}
        self.assertIn(1400.0, frequencies_at_120)
        self.assertIn(1300.0, frequencies_at_120)
        self.assertNotIn(3000.0, frequencies_at_120)

    def test_receiver_detects_pulse_from_environment_event(self):
        env = RadioEnvironment([
            make_record(toa_us=10.0, frequency_mhz=750.0, pulse_width_us=50.0, emitter_id=10)
        ])
        receiver = SieveReceiver(total_bandwidth=18e3, ibw=1e3, frequency_step=500.0, dwell_time=100.0, detection_threshold_db=-140.0)
        attach_receiver(env, receiver)
        env.run()
        self.assertTrue(any(d.detected for d in receiver.detection_history))


if __name__ == "__main__":
    unittest.main()

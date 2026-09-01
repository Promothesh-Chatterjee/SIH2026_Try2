import unittest

from src.environment.radio_environment import PulseRecord, RadioEnvironment
from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from src.models.random_scheduler import RandomScheduler
from src.receiver import SieveReceiver, attach_receiver


class TestRandomScheduler(unittest.TestCase):
    def test_random_scheduler_sees_receiver_observation_and_returns_valid_band(self):
        receiver = SieveReceiver(total_bandwidth=18000.0, ibw=1000.0, frequency_step=500.0, dwell_time=100.0)
        env = CognitiveRFScanEnv(receiver=receiver)
        radio_env = RadioEnvironment([
            PulseRecord(toa_us=10.0, frequency_mhz=750.0, pulse_width_us=50.0, amplitude_db=-120.0, aoa_deg=45.0, emitter_id=1),
            PulseRecord(toa_us=120.0, frequency_mhz=1400.0, pulse_width_us=50.0, amplitude_db=-118.0, aoa_deg=48.0, emitter_id=2),
        ])
        attach_receiver(radio_env, receiver)
        radio_env.run()

        obs = receiver.get_observation()
        scheduler = RandomScheduler(n_bands=8, seed=7)
        action = scheduler.act(obs)

        self.assertIsInstance(action, tuple)
        self.assertEqual(len(action), 2)
        self.assertIn(action[0], range(8))

    def test_random_scheduler_can_drive_env_without_training(self):
        receiver = SieveReceiver(total_bandwidth=18000.0, ibw=1000.0, frequency_step=500.0, dwell_time=100.0)
        env = CognitiveRFScanEnv(receiver=receiver)
        scheduler = RandomScheduler(n_bands=4, seed=42)

        obs = env.reset()
        action = scheduler.step(obs)
        self.assertIn(action, range(4))

        next_obs = env.step({"event": "entry", "time_us": 10.0, "pulse": {"frequency_mhz": 3000.0, "toa_us": 10.0, "pulse_width_us": 20.0, "amplitude_db": -110.0, "aoa_deg": 30.0, "pulse_id": 1, "emitter_id": 3}})
        self.assertIsNotNone(next_obs)


if __name__ == "__main__":
    unittest.main()

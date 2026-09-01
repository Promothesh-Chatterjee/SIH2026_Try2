import unittest

from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from src.environment.radio_environment import PulseRecord, RadioEnvironment
from src.models.random_scheduler import RandomScheduler


def _small_env(n_records=4, seed=7):
    cfg = {"n_bands": 18, "freq_min_mhz": 0.0, "freq_max_mhz": 18000.0, "ibw_mhz": 1000.0, "dwell_time_us": 500.0}
    records = [
        PulseRecord(0.0, 500.0, 20.0, 0.0, 10.0, emitter_id=0),
        PulseRecord(100.0, 500.0, 20.0, 8.0, 10.0, emitter_id=0),
        PulseRecord(4000.0, 9000.0, 10.0, 12.0, 30.0, emitter_id=1),
        PulseRecord(8000.0, 15000.0, 6.0, 9.0, 55.0, emitter_id=2),
    ][:n_records]
    return CognitiveRFScanEnv(cfg, records=records, seed=seed)


class TestRandomScheduler(unittest.TestCase):
    def test_radio_environment_emits_entry_and_exit_events(self):
        radio_env = RadioEnvironment([PulseRecord(10.0, 750.0, 50.0, -120.0, 45.0, emitter_id=1)])
        events = []
        while not radio_env.done:
            e = radio_env.step()
            if e is not None:
                events.append(e.event_type)
        self.assertEqual(events, ["entry", "exit"])

    def test_random_scheduler_drives_cognitive_env(self):
        env = _small_env()
        obs, _ = env.reset()
        scheduler = RandomScheduler(n_bands=env.action_space.n, seed=42)
        action = scheduler.step(obs)
        self.assertIn(action, range(env.action_space.n))

        next_obs, reward, term, trunc, info = env.step(action)
        self.assertIsNotNone(next_obs)
        self.assertTrue(env.observation_space.contains(next_obs))

    def test_cognitive_env_runs_end_to_end_with_dwell(self):
        env = _small_env(n_records=1)
        env.reset()
        # 500MHz pulse: band index = round(500/ (18000/18)) = band 0
        obs, reward, term, trunc, info = env.step(0)
        # Single pulse at 500MHz in band 0 -> intercepted
        self.assertTrue(info["hit"])
        self.assertEqual(len(info["detections"]), 1)


if __name__ == "__main__":
    unittest.main()
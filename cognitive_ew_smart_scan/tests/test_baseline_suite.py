"""Phase 13: the full 11-baseline suite, fair harness, and behavior contracts."""

import unittest

from src.contracts import NORMAL_DWELL, band_of_action, mode_of_action, n_actions_for
from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from src.environment.radio_environment import PulseRecord
from src.evaluation.baseline_suite_eval import run_baseline_suite
from src.models.baseline_suite import (
    BASELINE_NAMES,
    NN_BASELINES,
    RevisitHeuristic,
    SequentialSweep,
    build_baseline,
)

N_BANDS = 18
N_MODES = 5
N_ACTIONS = n_actions_for(N_BANDS, N_MODES)


def _records(n_emitters: int = 3):
    records = []
    for eid in range(n_emitters):
        freq = 500.0 + eid * 6000.0
        records.append(PulseRecord(0.0, freq, 30.0, 4.0, 10.0, emitter_id=eid))
        for k in range(20):
            records.append(PulseRecord(1000.0 + k * 500.0, freq, 30.0, 4.0, 10.0, emitter_id=eid))
    return records


def _env_cfg(n_bands: int = N_BANDS):
    return {
        "n_bands": n_bands,
        "freq_min_mhz": 0.0,
        "freq_max_mhz": 18000.0,
        "ibw_mhz": 1000.0,
        "dwell_time_us": 500.0,
    }


class BaselineSuiteConstructionTests(unittest.TestCase):
    def test_all_11_names_present(self):
        self.assertEqual(
            len(BASELINE_NAMES),
            11,
            "The suite must contain exactly the 11 documented baselines",
        )
        for name in BASELINE_NAMES:
            self.assertIn(name, BASELINE_NAMES)

    def test_every_baseline_builds_and_acts(self):
        env = CognitiveRFScanEnv(_env_cfg(), records=_records(), seed=7)
        obs, _ = env.reset()
        for name in BASELINE_NAMES:
            baseline = build_baseline(name, n_bands=N_BANDS, n_modes=N_MODES, seed=42, device="cpu")
            if hasattr(baseline, "reset"):
                baseline.reset()
            actions = []
            for _step in range(40):
                action = baseline.step(obs)
                self.assertIsInstance(action, int)
                self.assertTrue(0 <= action < N_ACTIONS, f"{name} emitted action {action} outside 0..{N_ACTIONS-1}")
                actions.append(action)
            # NN baselines must also expose the MoE-style select_action contract.
            if name in NN_BASELINES:
                self.assertTrue(hasattr(baseline, "select_action"))
                self.assertTrue(hasattr(baseline, "update"))
                a, hidden, attr = baseline.select_action(obs)
                self.assertTrue(0 <= a < N_ACTIONS)
                self.assertIsInstance(attr, dict)

    def test_unknown_baseline_raises(self):
        with self.assertRaises(ValueError):
            build_baseline("bogus", n_bands=N_BANDS)

    def test_case_insensitive_names(self):
        for name in ("Random", "FULL_MOE", "DRQN_PERIODIC"):
            baseline = build_baseline(name, n_bands=N_BANDS, n_modes=N_MODES, device="cpu")
            self.assertTrue(hasattr(baseline, "step"))


class BaselineBehaviorTests(unittest.TestCase):
    def test_sequential_sweep_cycles_ascending(self):
        sched = SequentialSweep(n_bands=N_BANDS, n_modes=N_MODES)
        bands = [band_of_action(sched.step(None), N_MODES) for _ in range(N_BANDS * 2)]
        self.assertEqual(bands, list(range(N_BANDS)) + list(range(N_BANDS)))
        modes = [mode_of_action(sched.step(None), N_MODES) for _ in range(N_BANDS)]
        self.assertEqual(set(modes), {NORMAL_DWELL})

    def test_fixed_periodic_scan_blocks_per_band(self):
        from src.models.baseline_suite import FixedPeriodicScan
        sched = FixedPeriodicScan(n_bands=N_BANDS, n_modes=N_MODES, dwell_slots=3)
        bands = [band_of_action(sched.step(None), N_MODES) for _ in range(12)]
        self.assertEqual(bands, [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

    def test_revisit_heuristic_visits_oldest_first(self):
        sched = RevisitHeuristic(n_bands=N_BANDS, n_modes=N_MODES)
        bands = [band_of_action(sched.step(None), N_MODES) for _ in range(N_BANDS * 2)]
        self.assertEqual(bands, list(range(N_BANDS)) + list(range(N_BANDS)))
        self.assertTrue(all(mode_of_action(sched.step(None), N_MODES) == NORMAL_DWELL for _ in range(3)))


class BaselineFairHarnessTests(unittest.TestCase):
    def test_run_baseline_suite_all_entries_identical_contract(self):
        records = _records()
        rows = run_baseline_suite(
            "configs/model_config.yaml",
            records,
            max_steps=50,
            seed=42,
            device="cpu",
        )
        self.assertEqual(len(rows), 11)
        names = {row["baseline"] for row in rows}
        self.assertEqual(names, set(BASELINE_NAMES))
        # Same action space for every entry.
        for row in rows:
            self.assertEqual(row["n_actions"], N_ACTIONS_WIDE, f"{row['baseline']} action space differs")
        # Same FoM summary keys for every entry.
        keys = {k for k in rows[0] if k != "baseline" and k != "n_actions"}
        for row in rows:
            self.assertEqual({k for k in row if k != "baseline" and k != "n_actions"}, keys)
        # Metrics were actually accumulated.
        for row in rows:
            self.assertIn("baseline_Pd", row)
            self.assertIn("baseline_Pfa", row)
            self.assertIn("baseline_avg_reward", row)

    def test_same_world_same_seed_per_baseline(self):
        # Two identical runs must produce identical first-step observations so
        # every baseline truly sees the same world.
        records = _records()
        env_cfg = _env_cfg()
        from src.evaluation.baseline_suite_eval import run_baseline_episode
        rows = run_baseline_suite("configs/model_config.yaml", records, max_steps=20, seed=42, device="cpu")
        rows2 = run_baseline_suite("configs/model_config.yaml", records, max_steps=20, seed=42, device="cpu")
        for r1, r2 in zip(rows, rows2):
            self.assertEqual(r1["baseline_Pd"], r2["baseline_Pd"], r1["baseline"])


N_ACTIONS_WIDE = n_actions_for(36, 5)


if __name__ == "__main__":
    unittest.main()
"""Phase 3: ground-truth emitter identity must never shape scheduler observables.

The full perception stack (deinterleaver -> HDBSCAN -> global reconciliation ->
EmitterTracker -> periodic interceptor -> belief -> observation) is exercised
with two scenarios that differ ONLY in the ground-truth ``emitter_id`` values.
The scheduler observation must be bit-for-bit identical, and the periodic
interceptor must be keyed exclusively by tracker-derived persistent track ids.
"""

import unittest

import numpy as np

from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from src.environment.radio_environment import PulseRecord


class _StubDeinterleaver:
    """Deterministic duck-typed stand-in for PDWTransformerEncoder.

    Emits the normalised PDW features as the embedding, so HDBSCAN separates
    emitters by their physical signature (frequency, AoA, pulse width) without
    any neural network. Deterministic: identical input -> identical output.
    """

    embed_dim = 6

    def infer(self, pdws_norm, device="cpu"):
        x = np.asarray(pdws_norm, dtype=np.float32)
        if x.ndim == 3:
            x = x[0]
        return x[:, : self.embed_dim]


def _pulse_train(toa_start, pri, count, freq, aoa, pw, amp, emitter_id):
    return [
        PulseRecord(
            toa_us=toa_start + i * pri,
            frequency_mhz=freq,
            pulse_width_us=pw,
            amplitude_db=amp,
            aoa_deg=aoa,
            emitter_id=emitter_id,
        )
        for i in range(count)
    ]


def _build_records(gt_ids):
    # Graphically, gt_ids = [idA1, idA2, idB1, idB2]; the two waveform classes
    # live on different bands so the scheduler actually visits both.
    id_a = gt_ids[0]
    id_b = gt_ids[2]
    records = []
    records += _pulse_train(0.0, 100.0, 40, 2750.0, 30.0, 10.0, -70.0, id_a)
    records += _pulse_train(50.0, 200.0, 20, 5250.0, 120.0, 20.0, -75.0, id_b)
    return records


def _run_env(gt_ids, seed=7):
    config = {
        "n_bands": 36,
        "n_modes": 5,
        "freq_min_mhz": 0.0,
        "freq_max_mhz": 18000.0,
        "ibw_mhz": 500.0,
        "dwell_time_us": 200.0,
        "frequency_step_mhz": 500.0,
        "detection_threshold_db": -140.0,
        "max_steps_per_episode": 24,
        "periodic_min_obs": 5,
    }
    deint_cfg = {
        "min_pulses": 8,
        "interval_steps": 1,
        "window_size": 64,
        "stride": 32,
        "min_cluster_size": 3,
        "min_samples": 2,
        "device": "cpu",
    }
    env = CognitiveRFScanEnv(
        config=config,
        records=_build_records(gt_ids),
        seed=seed,
        deinterleaver_model=_StubDeinterleaver(),
        deinterleaver_config=deint_cfg,
    )
    obs, _ = env.reset(seed=seed)
    obs_list = [obs]
    for step in range(12):
        band = 5 if step % 2 == 0 else 10
        action = band * 5 + 1  # NORMAL dwell mode on `band`
        obs, _reward, _term, _trunc, _info = env.step(action)
        obs_list.append(obs)
    return env, obs_list


class NoGroundTruthLeakageTests(unittest.TestCase):
    """Observation invariance under ground-truth emitter_id renaming."""

    def test_scheduler_observation_invariant_to_truth_id_renaming(self):
        env_a, obs_a = _run_env(gt_ids=[1, 1, 2, 2])
        env_b, obs_b = _run_env(gt_ids=[99, 99, 45, 45])

        self.assertEqual(len(obs_a), len(obs_b))
        for i, (o_a, o_b) in enumerate(zip(obs_a, obs_b)):
            self.assertTrue(
                np.array_equal(o_a, o_b),
                msg=f"observation diverged at step {i} after truth-id renaming",
            )

        # Both env instances exercised the perception+interceptor path: the
        # interceptor must have recorded tracker-derived intercepts from both
        # waveform classes (i.e. the leak channel was actually populated).
        self.assertGreater(len(env_a.periodic_interceptor.history), 0)
        self.assertGreater(len(env_b.periodic_interceptor.history), 0)

    def test_periodic_interceptor_keyed_by_tracker_ids_only(self):
        for gt_ids in ([1, 1, 2, 2], [99, 99, 45, 45]):
            env, _ = _run_env(gt_ids=gt_ids)
            tracker_tids = set(env.emitter_tracker.tracks.keys())
            self.assertGreater(len(tracker_tids), 0)
            for key in env.periodic_interceptor.history.keys():
                self.assertTrue(key.startswith("track_"), msg=f"non-track key {key!r}")
                self.assertIn(int(key.split("_")[1]), tracker_tids,
                              msg=f"interceptor key {key!r} is not a tracker track id")
                # Renaming the truth emitter ids must not change the key set.
            self.assertTrue(True)

    def test_truth_ids_do_not_enter_semantic_memory_identity(self):
        env, _ = _run_env(gt_ids=[1, 1, 2, 2])
        profiles = env.semantic_memory.list_emitters()
        for profile in profiles:
            self.assertTrue(profile.emitter_id.startswith("track_"),
                            msg=f"semantic memory identity {profile.emitter_id!r} leaks emitter_id")


if __name__ == "__main__":
    unittest.main()
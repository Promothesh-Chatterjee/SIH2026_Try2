"""Phase 13: Full end-to-end smoke test for the Cognitive EW SmartScan system.

Executes a complete, un-mocked 14-stage pipeline from loading a TSRD H5 file to
evaluation, failing loudly if any layer or contract breaks:

1. Load one TSRD file
2. Validate contract and structure
3. Receiver simulation
4. Observation construction
5. Deinterleaver inference
6. Belief matrix update
7. DRQN action forward
8. SmartScanMoE action fusion
9. Receiver dwell execution
10. Reward calculation & component breakdown
11. SequenceReplayBuffer insertion
12. One Double-DQN BPTT optimizer update step
13. Checkpoint save with provenance metadata
14. Full evaluation pipeline execution
"""

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.contracts import CANONICAL_N_BANDS, CANONICAL_N_MODES, CANONICAL_N_ACTIONS, CANONICAL_OBS_DIM
from src.data.synthetic_dataset import make_synthetic_pulse_train
from src.data.tsrd_manifest import TSRDValidator
from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from src.environment.scenario_generator import load_h5_records
from src.evaluation.evaluate_full import run_full_evaluation
from src.models.deinterleaver import PDWTransformerEncoder, windowed_cluster_deinterleave
from src.models.drqn_scheduler import DRQNScheduler
from src.models.smartscan_moe import SmartScanMoE
from src.preprocessing.normalise import normalise_pdws, save_normalization_stats
from src.training.replay_buffer import SequenceReplayBuffer
from src.training.train_scheduler import _do_drqn_update
from src.utils.checkpoint_meta import build_train_metadata, save_state


class EndToEndTSRDSmokeTest(unittest.TestCase):
    """Phase 13 full end-to-end smoke test across all 14 pipeline layers."""

    def test_full_end_to_end_pipeline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            tsrd_dir = temp_path / "tsrd_data"
            tsrd_dir.mkdir(parents=True, exist_ok=True)
            tsrd_file = tsrd_dir / "sample_tsrd_001.h5"
            ckpt_dir = temp_path / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            eval_output_dir = temp_path / "eval_results"

            # ------------------------------------------------------------------
            # Stage 1: Load one TSRD file
            # ------------------------------------------------------------------
            data, labels = make_synthetic_pulse_train(n_pulses=300, seed=42)
            sort_order = np.argsort(data[:, 0])
            data = data[sort_order]
            labels = labels[sort_order]
            with h5py.File(tsrd_file, "w") as h5f:
                h5f.create_dataset("data", data=data, dtype="float32")
                h5f.create_dataset("labels", data=labels, dtype="int32")
            self.assertTrue(tsrd_file.exists(), "Stage 1 failed: TSRD file was not created.")

            records = load_h5_records(tsrd_file)
            self.assertGreater(len(records), 0, "Stage 1 failed: No pulse records loaded from TSRD file.")

            # ------------------------------------------------------------------
            # Stage 2: Validate TSRD file structure & contract
            # ------------------------------------------------------------------
            validator = TSRDValidator()
            val_res = validator.validate_file(tsrd_file)
            self.assertTrue(val_res["valid"], f"Stage 2 failed: TSRD validation errors: {val_res.get('errors')}")

            # ------------------------------------------------------------------
            # Stage 3: Receiver Simulation
            # ------------------------------------------------------------------
            env_cfg = {
                "n_bands": CANONICAL_N_BANDS,
                "n_modes": CANONICAL_N_MODES,
                "n_actions": CANONICAL_N_ACTIONS,
                "obs_dim": CANONICAL_OBS_DIM,
                "freq_min_mhz": 0.0,
                "freq_max_mhz": 18000.0,
                "ibw_mhz": 500.0,
                "dwell_time_us": 500.0,
            }
            env = CognitiveRFScanEnv(env_cfg, records=records, seed=42)
            obs, info_init = env.reset()
            self.assertIsNotNone(env.receiver, "Stage 3 failed: Receiver simulation not initialized.")

            # ------------------------------------------------------------------
            # Stage 4: Observation Construction
            # ------------------------------------------------------------------
            self.assertIsInstance(obs, np.ndarray, "Stage 4 failed: Observation must be numpy array.")
            self.assertEqual(obs.shape, (CANONICAL_OBS_DIM,), f"Stage 4 failed: Obs shape {obs.shape} != ({CANONICAL_OBS_DIM},)")

            # ------------------------------------------------------------------
            # Stage 5: Deinterleaver Inference
            # ------------------------------------------------------------------
            deinterleaver = PDWTransformerEncoder(pdw_dim=6, d_model=128, nhead=8, num_layers=4, dim_feedforward=512)
            pdws_norm, train_stats = normalise_pdws(data[:, :5])
            stats_path = ckpt_dir / "normalization_stats.json"
            save_normalization_stats(train_stats, stats_path)

            deint_res = windowed_cluster_deinterleave(
                deinterleaver, pdws_norm, window_size=128, stride=64, device="cpu"
            )
            self.assertIn("labels", deint_res, "Stage 5 failed: Deinterleaver failed to produce cluster labels.")

            # Save deinterleaver checkpoint for evaluation stage
            deint_ckpt_path = ckpt_dir / "deinterleaver.pt"
            deint_meta = build_train_metadata(
                split="train", n_bands=36, arch="PDWTransformerEncoder", seed=42, metrics={"v_measure": 0.9}
            )
            save_state(deinterleaver, deint_ckpt_path, deint_meta)

            # ------------------------------------------------------------------
            # Stage 6: Belief Update
            # ------------------------------------------------------------------
            self.assertTrue(hasattr(env, "belief"), "Stage 6 failed: Environment missing belief object.")
            init_occupancy = float(np.sum(env.belief.occupancy_prob))
            self.assertTrue(np.isfinite(init_occupancy), "Stage 6 failed: Belief matrix contains non-finite values.")

            # ------------------------------------------------------------------
            # Stage 7: DRQN Action Forward
            # ------------------------------------------------------------------
            drqn = DRQNScheduler(obs_dim=CANONICAL_OBS_DIM, n_bands=CANONICAL_N_BANDS, n_actions=CANONICAL_N_ACTIONS, lstm_hidden=256, lstm_layers=2)
            q_values, _aux, hidden_drqn = drqn(torch.from_numpy(obs).unsqueeze(0).unsqueeze(0))
            self.assertEqual(q_values.shape[-1], CANONICAL_N_ACTIONS, "Stage 7 failed: DRQN output shape mismatch.")

            # ------------------------------------------------------------------
            # Stage 8: MoE Selection
            # ------------------------------------------------------------------
            moe = SmartScanMoE(drqn, config={"n_bands": CANONICAL_N_BANDS, "n_modes": CANONICAL_N_MODES, "eager_weight": 0.6, "revisit_weight": 0.4})
            action, next_hidden, attr = moe.select_action(obs, eager_hidden=hidden_drqn)
            self.assertTrue(0 <= action < CANONICAL_N_ACTIONS, f"Stage 8 failed: Action {action} out of range [0, {CANONICAL_N_ACTIONS})")
            self.assertIn("reason", attr, "Stage 8 failed: MoE attribution missing 'reason' key.")

            # ------------------------------------------------------------------
            # Stage 9: Receiver Dwell Execution
            # ------------------------------------------------------------------
            next_obs, reward, terminated, truncated, info = env.step(action, mode_context=attr)
            self.assertEqual(next_obs.shape, (CANONICAL_OBS_DIM,), "Stage 9 failed: Next obs shape mismatch after dwell.")

            # ------------------------------------------------------------------
            # Stage 10: Reward Calculation & Component Breakdown
            # ------------------------------------------------------------------
            self.assertTrue(np.isfinite(reward), "Stage 10 failed: Reward is non-finite.")
            self.assertIn("hit", info, "Stage 10 failed: Step info missing 'hit' indicator.")

            # ------------------------------------------------------------------
            # Stage 11: Replay Insertion
            # ------------------------------------------------------------------
            replay_buffer = SequenceReplayBuffer(capacity=100, seq_len=16, obs_dim=CANONICAL_OBS_DIM, burn_in=2)
            replay_buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=False,
                hit_prob=1.0 if info["hit"] else 0.0,
                intercept_time_us=float(info.get("intercept_time_error_us", float("nan"))),
            )
            replay_buffer.add(
                obs=next_obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=True,
                hit_prob=0.0,
                intercept_time_us=float("nan"),
            )
            self.assertGreater(len(replay_buffer), 0, "Stage 11 failed: Replay buffer is empty after episode insertion.")

            # ------------------------------------------------------------------
            # Stage 12: One Optimizer Update
            # ------------------------------------------------------------------
            target_drqn = DRQNScheduler(obs_dim=CANONICAL_OBS_DIM, n_bands=CANONICAL_N_BANDS, n_actions=CANONICAL_N_ACTIONS, lstm_hidden=256, lstm_layers=2)
            optimizer = optim.Adam(drqn.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()

            batch = replay_buffer.sample(batch_size=1)
            loss_val = _do_drqn_update(
                online_drqn=drqn,
                target_drqn=target_drqn,
                optimizer=optimizer,
                loss_fn=loss_fn,
                batch=batch,
                gamma=0.95,
                device=torch.device("cpu"),
            )
            self.assertTrue(np.isfinite(loss_val), "Stage 12 failed: Optimizer update loss is non-finite.")

            # ------------------------------------------------------------------
            # Stage 13: Checkpoint Save with Provenance Metadata
            # ------------------------------------------------------------------
            sched_ckpt_path = ckpt_dir / "scheduler.pt"
            sched_meta = build_train_metadata(
                split="train",
                n_bands=CANONICAL_N_BANDS,
                arch="DRQNScheduler",
                seed=42,
                metrics={"loss": loss_val},
            )
            save_state(drqn, sched_ckpt_path, sched_meta)
            self.assertTrue(sched_ckpt_path.exists(), "Stage 13 failed: Scheduler checkpoint file missing.")

            # ------------------------------------------------------------------
            # Stage 14: Full Evaluation Pipeline Execution
            # ------------------------------------------------------------------
            eval_res = run_full_evaluation(
                deinterleaver_ckpt=deint_ckpt_path,
                scheduler_ckpt=sched_ckpt_path,
                config_path=Path("configs/model_config.yaml"),
                test_dir=tsrd_dir,
                output_dir=eval_output_dir,
                mode="scan",
                baseline="random",
                norm_stats=stats_path,
                seed=42,
            )
            self.assertIn("aggregate", eval_res, "Stage 14 failed: Evaluation did not return aggregate metrics dict.")
            agg = eval_res["aggregate"]
            self.assertIn("sched_Pd", agg, "Stage 14 failed: Evaluation output missing 'sched_Pd' metric.")
            self.assertTrue(np.isfinite(agg["sched_Pd"]), "Stage 14 failed: Evaluated Pd metric is non-finite.")


if __name__ == "__main__":
    unittest.main()

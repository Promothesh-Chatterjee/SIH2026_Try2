import numpy as np
import pytest
import torch

from ew_core.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from ew_core.environment.radio_environment import PulseRecord
from ew_core.models.drqn_scheduler import DRQNScheduler


class _DeterministicDeinterleaver:
    """Deterministic mock deinterleaver that projects physics features into embedding."""
    embed_dim = 6

    def infer(self, pdws_norm, device="cpu"):
        x = np.asarray(pdws_norm, dtype=np.float32)
        if x.ndim == 3:
            x = x[0]
        return x[:, : self.embed_dim]


def _generate_pulse_train(toa_start, pri, count, freq, aoa, pw, amp, emitter_id):
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


def _build_test_scenario(emitter_ids: list[int]):
    records = []
    # Emitter 1 in Band 5 (2500 - 3000 MHz)
    records += _generate_pulse_train(
        toa_start=10.0, pri=100.0, count=50,
        freq=2750.0, aoa=45.0, pw=12.0, amp=-70.0,
        emitter_id=emitter_ids[0]
    )
    # Emitter 2 in Band 12 (6000 - 6500 MHz)
    records += _generate_pulse_train(
        toa_start=25.0, pri=150.0, count=40,
        freq=6250.0, aoa=120.0, pw=25.0, amp=-75.0,
        emitter_id=emitter_ids[1]
    )
    records.sort(key=lambda r: r.toa_us)
    return records


def test_causal_full_chain_zero_gt_leakage():
    """Execute end-to-end integration pipeline:
    TSRD -> Receiver -> PDW -> Deinterleaver -> Tracker -> Belief -> Obs -> Model -> Action -> Reward
    Demonstrates ZERO Ground Truth (emitter_id) leakage across the entire operational chain.
    """
    config = {
        "n_bands": 36,
        "n_modes": 5,
        "freq_min_mhz": 0.0,
        "freq_max_mhz": 18000.0,
        "ibw_mhz": 500.0,
        "dwell_time_us": 200.0,
        "frequency_step_mhz": 500.0,
        "detection_threshold_db": -110.0,
        "max_steps_per_episode": 20,
        "periodic_min_obs": 3,
        "receiver": {
            "sensitivity_dbm": -110.0,
            "cfar_guard_cells": 2,
            "cfar_ref_cells": 8,
            "cfar_pfa": 0.001,
            "noise_floor_isolation": True,
        },
    }
    deint_cfg = {
        "min_pulses": 6,
        "interval_steps": 1,
        "window_size": 64,
        "stride": 32,
        "min_cluster_size": 3,
        "min_samples": 2,
        "device": "cpu",
    }

    # Model for action selection
    torch.manual_seed(42)
    model = DRQNScheduler(
        obs_dim=360,
        n_bands=36,
        n_modes=5,
        n_actions=180,
        lstm_hidden=128,
        lstm_layers=2,
    )
    model.eval()

    # Run 1: with ground truth emitter IDs [101, 202]
    records_1 = _build_test_scenario([101, 202])
    env_1 = CognitiveRFScanEnv(
        config=config,
        records=records_1,
        seed=123,
        deinterleaver_model=_DeterministicDeinterleaver(),
        deinterleaver_config=deint_cfg,
    )

    # Run 2: with completely different / scrambled ground truth emitter IDs [-9999, 88888]
    records_2 = _build_test_scenario([-9999, 88888])
    env_2 = CognitiveRFScanEnv(
        config=config,
        records=records_2,
        seed=123,
        deinterleaver_model=_DeterministicDeinterleaver(),
        deinterleaver_config=deint_cfg,
    )

    obs1, _ = env_1.reset(seed=123)
    obs2, _ = env_2.reset(seed=123)

    np.testing.assert_array_equal(obs1, obs2, err_msg="Initial observations diverged despite identical RF physics!")

    n_steps = 15
    h1, h2 = None, None
    for s in range(n_steps):
        # Neural Model inference from observation
        with torch.no_grad():
            t_obs1 = torch.as_tensor(obs1, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            t_obs2 = torch.as_tensor(obs2, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            q1_t, aux1, h1 = model(t_obs1, h1)
            q2_t, aux2, h2 = model(t_obs2, h2)

            q1 = q1_t[0, 0].numpy()
            q2 = q2_t[0, 0].numpy()

            np.testing.assert_allclose(q1, q2, atol=1e-6, err_msg=f"Step {s}: Q-values diverged!")

            action1 = int(np.argmax(q1))
            action2 = int(np.argmax(q2))
            assert action1 == action2, f"Step {s}: Action mismatch {action1} vs {action2}"

        # Step environments
        next_obs1, r1, term1, trunc1, info1 = env_1.step(action1)
        next_obs2, r2, term2, trunc2, info2 = env_2.step(action2)

        # 1. Observation contract and bit-level invariance
        assert len(next_obs1) == 360
        assert len(next_obs2) == 360
        np.testing.assert_array_equal(
            next_obs1, next_obs2,
            err_msg=f"Step {s}: Observations diverged under GT ID perturbation!"
        )

        # 2. Reward calculation invariance
        assert abs(r1 - r2) < 1e-9, f"Step {s}: Rewards diverged ({r1} vs {r2})!"

        # 3. Termination invariance
        assert term1 == term2
        assert trunc1 == trunc2

        obs1 = next_obs1
        obs2 = next_obs2

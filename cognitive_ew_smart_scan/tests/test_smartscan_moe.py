"""Phase 11: SmartScanMoE comprehensive test suite.

Verifies:
1. Deterministic action selection under deterministic inputs.
2. Band-to-action broadcasting alignment across the 180-action space (band * n_modes + mode).
3. Revisit urgency tracking and max_revisit_gap thresholding.
4. Periodic urgency vector integration.
5. Q-value dominance over default semantic mode intent scores.
6. Config-driven fusion weights.
7. Recurrent LSTM hidden state single-step progression.
8. Parity between PyTorch batched forward() and single-step select_action().
"""

import unittest
import numpy as np
import torch

from src.contracts import (
    CANONICAL_N_BANDS,
    CANONICAL_N_MODES,
    CANONICAL_N_ACTIONS,
    band_of_action,
    mode_of_action,
    encode_action,
)
from src.models.drqn_scheduler import DRQNScheduler
from src.models.smartscan_moe import SmartScanMoE


def _create_moe(config: dict | None = None) -> SmartScanMoE:
    drqn = DRQNScheduler(obs_dim=360, n_bands=36, n_actions=180, lstm_hidden=32, lstm_layers=1)
    cfg = {
        "n_bands": 36,
        "n_modes": 5,
        "n_actions": 180,
        "eager_weight": 0.6,
        "revisit_weight": 0.4,
        "preemptive_weight": 0.0,
        "semantic_weight": 0.1,
    }
    if config:
        cfg.update(config)
    return SmartScanMoE(drqn, config=cfg)


def _sample_obs() -> np.ndarray:
    rng = np.random.RandomState(42)
    return rng.uniform(0.0, 1.0, 360).astype(np.float32)


class SmartScanMoETests(unittest.TestCase):
    """Phase 11 SmartScanMoE audit and verification tests."""

    def test_deterministic_selection(self):
        """Deterministic inputs must produce 100% deterministic selections."""
        moe = _create_moe()
        obs = _sample_obs()
        moe.eager_agent.reset(batch_size=1)
        init_h = moe.eager_agent.hidden

        # Call select_action twice with same obs and explicit hidden state
        act1, h1, attr1 = moe.select_action(obs, eager_hidden=init_h)
        act2, h2, attr2 = moe.select_action(obs, eager_hidden=init_h)

        self.assertEqual(act1, act2)
        self.assertAlmostEqual(attr1["action_score"], attr2["action_score"])
        self.assertEqual(attr1["selected_band"], attr2["selected_band"])
        self.assertEqual(attr1["selected_mode"], attr2["selected_mode"])

        # Top-k select_bands determinism
        bands1, _, attr_b1 = moe.select_bands(obs, eager_hidden=init_h, k=3, return_full=True)
        bands2, _, attr_b2 = moe.select_bands(obs, eager_hidden=init_h, k=3, return_full=True)
        self.assertEqual(bands1, bands2)
        self.assertAlmostEqual(attr_b1["eager_pct"], attr_b2["eager_pct"])

    def test_band_action_broadcasting(self):
        """Urgency per band must broadcast identically across all n_modes for that band."""
        moe = _create_moe({"eager_weight": 0.0, "revisit_weight": 1.0, "semantic_weight": 0.0})
        # Simulate visits: band 10 visited recently (t=10), band 5 visited at t=0 (current_t=10)
        moe.revisit_agent.last_visit_time[10] = 10.0
        moe.revisit_agent.last_visit_time[5] = 0.0
        moe.revisit_agent.current_t = 10

        scores = moe.revisit_agent.action_scores(CANONICAL_N_MODES)
        self.assertEqual(len(scores), CANONICAL_N_ACTIONS)

        # All 5 modes for band 5 must have equal urgency score
        b5_scores = [scores[encode_action(5, m)] for m in range(CANONICAL_N_MODES)]
        self.assertEqual(len(set(b5_scores)), 1)

        # Band 5 (oldest visit) must have higher urgency than Band 10 (recent visit)
        b10_scores = [scores[encode_action(10, m)] for m in range(CANONICAL_N_MODES)]
        self.assertGreater(b5_scores[0], b10_scores[0])

    def test_revisit_urgency_and_max_gap(self):
        """Revisit urgency increases with elapsed time and caps at 1.0 when exceeding max_revisit_gap."""
        moe = _create_moe({"max_revisit_gap": 50})

        # Band 0 visited at t=0, current_t=60 (> max_revisit_gap 50)
        moe.revisit_agent.last_visit_time[0] = 0.0
        moe.revisit_agent.current_t = 60
        scores = moe.revisit_agent.scores()

        self.assertEqual(scores[0], 1.0)  # Overdue gap boost enforced

    def test_periodic_urgency_vector(self):
        """Setting periodic urgency vector boosts targeted band preemptive action."""
        moe = _create_moe({"eager_weight": 0.0, "revisit_weight": 0.0, "preemptive_weight": 1.0, "semantic_weight": 0.0})
        vec = np.zeros(CANONICAL_N_BANDS, dtype=np.float32)
        vec[12] = 0.95
        moe.set_periodic_urgency_vector(vec)

        act, _, attr = moe.select_action(_sample_obs())
        self.assertEqual(band_of_action(act), 12)
        self.assertAlmostEqual(attr["periodic_urgency"], 0.95)

    def test_semantic_scores_do_not_dominate_q(self):
        """Strong DRQN Q-value overrides default heuristic semantic scores."""
        moe = _create_moe({"eager_weight": 1.0, "revisit_weight": 0.0, "semantic_weight": 0.1})

        # Mock DRQN get_q output
        target_action = encode_action(20, 0)  # Band 20, SHORT_DWELL
        q_raw = np.zeros(CANONICAL_N_ACTIONS, dtype=np.float32)
        q_raw[target_action] = 10.0  # Dominant Q value

        # Override eager_agent.get_q to return controlled Q values
        moe.eager_agent.get_q = lambda obs, hidden=None: (q_raw, hidden)

        obs = np.zeros(360, dtype=np.float32)
        act, _, _ = moe.select_action(obs)

        self.assertEqual(act, target_action)

    def test_config_driven_fusion_weights(self):
        """All fusion weights and hyperparameters must be driven by config dict."""
        cfg = {
            "eager_weight": 0.75,
            "revisit_weight": 0.25,
            "preemptive_weight": 0.15,
            "semantic_weight": 0.05,
            "k_receivers": 4,
            "decay_rate": 0.08,
            "max_revisit_gap": 150,
        }
        moe = _create_moe(cfg)

        self.assertEqual(moe.eager_weight, 0.75)
        self.assertEqual(moe.revisit_weight, 0.25)
        self.assertEqual(moe.preemptive_weight, 0.15)
        self.assertEqual(moe.semantic_weight, 0.05)
        self.assertEqual(moe.k_receivers, 4)
        self.assertEqual(moe.decay_rate, 0.08)
        self.assertEqual(moe.max_revisit_gap, 150)
        self.assertEqual(moe.revisit_agent.decay_rate, 0.08)
        self.assertEqual(moe.revisit_agent.max_revisit_gap, 150)

    def test_recurrent_state_single_step(self):
        """Recurrent LSTM state must advance exactly once per select_action call."""
        moe = _create_moe()
        moe.eager_agent.reset(batch_size=1)
        init_h = moe.eager_agent.hidden

        obs = _sample_obs()
        act, next_h, _ = moe.select_action(obs, eager_hidden=init_h)

        self.assertIsNotNone(next_h)
        # Check that hidden tensor actually changed
        self.assertFalse(torch.equal(init_h[0], next_h[0]))

    def test_forward_and_select_action_parity(self):
        """PyTorch batched forward() must align with single-step select_action fused scores."""
        moe = _create_moe({"eager_weight": 0.6, "revisit_weight": 0.0, "semantic_weight": 0.1})
        moe.eager_agent.reset()

        obs_np = _sample_obs()
        fused_np, eager_norm, revisit_norm, _, _ = moe._compute_fused(obs_np)

        obs_t = torch.from_numpy(obs_np).unsqueeze(0).unsqueeze(0)  # (1, 1, 360)
        fused_t, _, _ = moe.forward(obs_t)
        fused_t_np = fused_t[0, 0].detach().cpu().numpy()

        # Neural & semantic score fusion parity
        np.testing.assert_allclose(fused_np, fused_t_np, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()

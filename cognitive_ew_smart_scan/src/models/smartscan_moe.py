"""
SmartScan Mixture of Experts (MoE) with EagerAgent + RevisitAgent.

EagerAgent wraps DRQN + min-max norm; RevisitAgent is algorithmic exp decay.
Fused = eager_weight * eager_norm + revisit_weight * revisit_norm; top-K selection.

The MoE operates over the canonical time-frequency joint action space:
``action = band * n_modes + mode`` (flat index into ``n_bands*n_modes``). The
RevisitAgent's per-band urgency is broadcast across all dwell modes of each band so
that visit pressure applies to the whole time-frequency cell of a band.
"""

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.contracts import (
    CANONICAL_N_BANDS,
    CANONICAL_N_MODES,
    DEFAULT_DWELL_MULTIPLIERS,
    DWELL_MODES,
    DWELL_MODE_SEMANTICS,
    OCCUPANCY_IDX,
    REVISIT_AGE_IDX,
    UNCERTAINTY_IDX,
    SHORT_DWELL,
    NORMAL_DWELL,
    LONG_DWELL,
    REVISIT,
    PREEMPTIVE_INTERCEPT,
    band_of_action,
    mode_of_action,
    n_actions_for,
)
from .drqn_scheduler import DRQNScheduler

logger = logging.getLogger(__name__)


class SmartScanMoE(nn.Module):
    """Mixture of Experts fusing DRQN and revisitation heuristic over time-frequency.

    Contains inner classes EagerAgent and RevisitAgent per spec.
    """

    class EagerAgent:
        """Wraps DRQN, returns normalised Q-values; maintains LSTM hidden state."""

        def __init__(self, drqn: DRQNScheduler, device: torch.device | str = "cpu") -> None:
            """Initialise eager agent.

            Args:
                drqn: DRQN scheduler instance.
                device: Device for inference.
            """
            self.drqn = drqn
            self.device = torch.device(device) if isinstance(device, str) else device
            self.hidden: tuple[torch.Tensor, torch.Tensor] | None = None
            self.n_bands = drqn.n_bands
            self.n_actions = drqn.n_actions

        def reset(self, batch_size: int = 1) -> None:
            """Reset LSTM hidden state.

            Args:
                batch_size: Batch size for hidden init.
            """
            try:
                self.hidden = self.drqn.init_hidden(batch_size, self.device)
            except Exception:
                self.hidden = None

        def get_q(self, obs: torch.Tensor) -> tuple[np.ndarray, tuple[torch.Tensor, torch.Tensor] | None]:
            """Run DRQN forward, return Q-values and next hidden.

            Args:
                obs: (B,T,obs_dim) or (obs_dim,) tensor.

            Returns:
                Tuple (q_values_np (n_actions,), hidden).
            """
            # Ensure (1,1,obs_dim) for single step
            if obs.dim() == 1:
                obs_b = obs.unsqueeze(0).unsqueeze(0)
            elif obs.dim() == 2:
                obs_b = obs.unsqueeze(1)
            else:
                obs_b = obs
            obs_b = obs_b.to(self.device)
            try:
                self.drqn.to(self.device)
            except Exception:
                pass
            with torch.inference_mode():
                q, _aux, h = self.drqn(obs_b, self.hidden)
                self.hidden = h
                # Last timestep, first batch
                q_last = q[0, -1].detach().cpu().numpy()
            return q_last, h

        def normalised_scores(self, q_values: np.ndarray) -> np.ndarray:
            """Min-max normalise Q-values to [0,1].

            Args:
                q_values: (n_actions,) raw Q.

            Returns:
                (n_actions,) normalised.
            """
            q_min = float(np.min(q_values))
            q_max = float(np.max(q_values))
            if q_max - q_min < 1e-8:
                return np.zeros_like(q_values, dtype=np.float32)
            return ((q_values - q_min) / (q_max - q_min + 1e-8)).astype(np.float32)

    class RevisitAgent:
        """Algorithmic revisit urgency: exp(decay_rate * (t - last_visit)).

        Ensures no band ignored > max_revisit_gap slots.
        """

        def __init__(self, n_bands: int = CANONICAL_N_BANDS, decay_rate: float = 0.05, max_revisit_gap: int | None = None) -> None:
            """Initialise revisit agent.

            Args:
                n_bands: Number of bands.
                decay_rate: Exponential rate.
                max_revisit_gap: Max slots before forced revisit.
            """
            self.n_bands = n_bands
            self.decay_rate = decay_rate
            self.max_revisit_gap = int(max_revisit_gap if max_revisit_gap is not None else 200)
            self.last_visit_time = np.zeros(n_bands, dtype=np.float64)
            self.current_t: int = 0

        def update(self, selected_band: int) -> None:
            """Update timestamp for selected band.

            Args:
                selected_band: Band tuned this step.
            """
            self.last_visit_time[int(selected_band)] = float(self.current_t)
            self.current_t += 1

        def scores(self) -> np.ndarray:
            """Return per-band urgency vector exp(decay * (t - last_visit)), normalised to [0,1].

            Returns:
                (n_bands,) float32 in [0,1].
            """
            raw = np.exp(self.decay_rate * (self.current_t - self.last_visit_time))
            # Min-max to [0,1] for fusion
            r_min, r_max = float(np.min(raw)), float(np.max(raw))
            if r_max - r_min < 1e-8:
                return np.zeros_like(raw, dtype=np.float32)
            norm = (raw - r_min) / (r_max - r_min + 1e-8)
            # Enforce max gap: if any band exceeds gap, boost to 1
            overdue = (self.current_t - self.last_visit_time) > self.max_revisit_gap
            norm[overdue] = 1.0
            return norm.astype(np.float32)

        def action_scores(self, n_modes: int) -> np.ndarray:
            """Broadcast per-band urgency across all dwell modes of each band.

            Args:
                n_modes: Number of dwell modes.

            Returns:
                (n_bands * n_modes,) float32 urgency per time-frequency action.
            """
            per_band = self.scores()  # (n_bands,)
            return np.repeat(per_band, n_modes).astype(np.float32)

        def reset(self) -> None:
            """Reset visit times."""
            self.last_visit_time.fill(0.0)
            self.current_t = 0

    def __init__(self, drqn_agent: DRQNScheduler, config: dict[str, Any] | None = None) -> None:
        """Initialise MoE fusion.

        Args:
            drqn_agent: DRQN scheduler instance.
            config: Dict with eager_weight, revisit_weight, k_receivers, decay_rate,
                n_bands, n_modes.
        """
        super().__init__()
        config = config or {}
        self.n_bands: int = int(config.get("n_bands", getattr(drqn_agent, "n_bands", CANONICAL_N_BANDS)))
        self.n_modes: int = int(config.get("n_modes", getattr(drqn_agent, "n_modes", CANONICAL_N_MODES)))
        self.n_actions: int = int(config.get("n_actions", n_actions_for(self.n_bands, self.n_modes)))
        self.eager_weight: float = float(config.get("eager_weight", 0.6))
        self.revisit_weight: float = float(config.get("revisit_weight", 0.4))
        self.preemptive_weight: float = float(config.get("preemptive_weight", 0.0))
        self.semantic_weight: float = float(config.get("semantic_weight", 1.0))
        self.k_receivers: int = int(config.get("k_receivers", 1))
        self.decay_rate: float = float(config.get("decay_rate", 0.05))
        self.max_revisit_gap: int = int(config.get("max_revisit_gap", 200))

        # Inner agents per spec
        device = config.get("device", "cpu")
        self.eager_agent = SmartScanMoE.EagerAgent(drqn_agent, device=device)
        self.revisit_agent = SmartScanMoE.RevisitAgent(
            n_bands=self.n_bands,
            decay_rate=self.decay_rate,
            max_revisit_gap=self.max_revisit_gap,
        )

        # Periodic preemptive prioritisation map: band -> urgency boost.
        self._preemptive_urgency = np.zeros(self.n_bands, dtype=np.float32)

        # Keep direct refs for torch MoE forward
        self.drqn = drqn_agent
        self._config = config
        logger.info(
            "SmartScanMoE eager=%.1f revisit=%.1f preemptive=%.1f K=%d actions=%d",
            self.eager_weight, self.revisit_weight, self.preemptive_weight,
            self.k_receivers, self.n_actions,
        )

    def set_preemptive_urgency(self, band: int | None, urgency: float) -> None:
        """Fold a periodic-intercept urgency boost into a band's selection pressure.

        Args:
            band: Band index, or None to clear.
            urgency: Urgency in [0, 1].
        """
        if band is not None and 0 <= int(band) < self.n_bands:
            self._preemptive_urgency[int(band)] = float(np.clip(urgency, 0.0, 1.0))

    def set_periodic_urgency_vector(self, urgency: np.ndarray | list | tuple) -> None:
        """Set the full per-band periodic-/preemptive-urgency vector.

        Args:
            urgency: Real-valued array-like of length n_bands, clipped to [0, 1].
        """
        vec = np.asarray(urgency, dtype=np.float32).reshape(-1)
        if vec.size != self.n_bands:
            raise ValueError(f"periodic urgency vector size {vec.size} != n_bands {self.n_bands}")
        self._preemptive_urgency[:] = np.clip(vec, 0.0, 1.0)

    def _mode_semantic_scores(self, obs_1d: np.ndarray) -> np.ndarray:
        """Per-(band, mode) semantic intent scores from one observation vector.

        Each dwell mode encodes a distinct *reason* to act (not just dwell length):

          - SHORT_DWELL        recce: quick look that loses to any urgent pressure
          - NORMAL_DWELL       surveillance: neutral default
          - LONG_DWELL         deep observation of band(s) with high uncertainty
          - REVISIT            driven by revisit age (urges overdue bands)
          - PREEMPTIVE_INTERCEPT  driven by periodic-imminent-arrival urgency

        Returns:
            (n_bands, n_modes) float32 scores in [0, 1].
        """
        scores = np.zeros((self.n_bands, self.n_modes), dtype=np.float32)
        if obs_1d is None or obs_1d.size == 0:
            scores[:, NORMAL_DWELL] = 0.45
            return scores
        fpb = max(1, int(obs_1d.size) // self.n_bands)
        occ = np.clip(obs_1d[OCCUPANCY_IDX::fpb][: self.n_bands], 0.0, 1.0)
        unc = np.clip(obs_1d[UNCERTAINTY_IDX::fpb][: self.n_bands], 0.0, 1.0)
        rev = np.clip(obs_1d[REVISIT_AGE_IDX::fpb][: self.n_bands], 0.0, 1.0)
        per = np.clip(self._preemptive_urgency[: self.n_bands], 0.0, 1.0)
        peak = np.maximum(rev, per)
        scores[:, SHORT_DWELL] = 0.30 * (1.0 - peak)
        scores[:, NORMAL_DWELL] = 0.45
        scores[:, LONG_DWELL] = 0.25 + 0.35 * unc
        scores[:, REVISIT] = 0.20 + 0.60 * rev
        scores[:, PREEMPTIVE_INTERCEPT] = 0.20 + 0.60 * per
        return scores

    def _compute_fused(
        self, obs: np.ndarray | torch.Tensor, eager_hidden: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[torch.Tensor, torch.Tensor] | None, np.ndarray]:
        """Compute fused per-action scores for numpy or torch observations.

        Returns:
            Tuple (fused (n_actions,), eager_norm (n_actions,), revisit_norm (n_bands,),
            hidden, obs_1d (obs_dim,) float32).
        """
        if isinstance(obs, np.ndarray):
            obs_1d = np.asarray(obs).reshape(-1).astype(np.float32)
            obs_t = torch.from_numpy(obs_1d)
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0).unsqueeze(0)
            elif obs_t.dim() == 2:
                obs_t = obs_t.unsqueeze(1)
            q_raw, hidden = self.eager_agent.get_q(obs_t.squeeze(0) if obs_t.shape[0] == 1 else obs_t)
            eager_norm = self.eager_agent.normalised_scores(q_raw)
        elif isinstance(obs, torch.Tensor):
            obs_1d = obs[0, -1].detach().cpu().numpy().reshape(-1)
            q, _aux, hidden = self.drqn(obs, eager_hidden)
            self.eager_agent.hidden = hidden
            q_last = q[0, -1].detach().cpu().numpy() if q.dim() == 3 else q.detach().cpu().numpy()
            if q_last.ndim > 1:
                q_last = q_last[0]
            eager_norm = self.eager_agent.normalised_scores(q_last)
        else:
            raise TypeError(f"Unsupported obs type {type(obs)}")

        revisit_norm = self.revisit_agent.scores()
        semantic = self._mode_semantic_scores(obs_1d)
        fused = self._fused_action_scores(eager_norm, revisit_norm, semantic)
        return fused, eager_norm, revisit_norm, hidden if "hidden" in locals() else eager_hidden, obs_1d

    def _fused_action_scores(
        self, eager_norm: np.ndarray, revisit_norm_per_band: np.ndarray,
        semantic_scores: np.ndarray | None = None,
    ) -> np.ndarray:
        """Combine eager, revisit, preemptive and mode-semantic terms into per-action fused scores."""
        revisit_action = np.repeat(revisit_norm_per_band, self.n_modes).astype(np.float32)
        fused = self.eager_weight * eager_norm + self.revisit_weight * revisit_action
        # Periodic preemptive pressure: parent band urgency broadcast across its modes.
        preempt_action = np.repeat(self._preemptive_urgency, self.n_modes).astype(np.float32)
        fused = fused + self.preemptive_weight * preempt_action
        if semantic_scores is not None:
            fused = fused + self.semantic_weight * np.asarray(semantic_scores, dtype=np.float32).reshape(-1)
        return fused

    def _attribution_for(self, action: int, obs_1d: np.ndarray) -> dict[str, float | int | str]:
        """Explainability record: WHY the chosen time-frequency action was selected.

        The mode's semantic reason (see DWELL_MODE_SEMANTICS) plus the numeric
        urgency drivers let the action-selection layer distinguish e.g. a REVISIT
        (driven by revisit age) from a PREEMPTIVE_INTERCEPT (driven by a predicted
        periodic arrival).
        """
        band = band_of_action(int(action), self.n_modes)
        mode = mode_of_action(int(action), self.n_modes)
        fpb = max(1, int(obs_1d.size) // self.n_bands) if obs_1d is not None and obs_1d.size else 10
        if obs_1d is not None and obs_1d.size > 0:
            rev = float(np.clip(obs_1d[REVISIT_AGE_IDX::fpb][band], 0.0, 1.0))
            unc = float(np.clip(obs_1d[UNCERTAINTY_IDX::fpb][band], 0.0, 1.0))
        else:
            rev = 0.0
            unc = 0.0
        per = float(np.clip(self._preemptive_urgency[band], 0.0, 1.0))
        return {
            "selected_band": band,
            "selected_mode": int(mode),
            "mode_name": DWELL_MODES[int(mode)],
            "reason": DWELL_MODE_SEMANTICS[int(mode)],
            "revisit_urgency": rev,
            "periodic_urgency": per,
            "uncertainty_urgency": unc,
        }

    def select_action(
        self, obs: np.ndarray | torch.Tensor, eager_hidden: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[int, tuple[torch.Tensor, torch.Tensor] | None, dict[str, float]]:
        """Select the single best time-frequency action (band, dwell-mode).

        Args:
            obs: Observation vector (obs_dim,) numpy or (B,T,obs_dim) torch.
            eager_hidden: Optional LSTM hidden state.

        Returns:
            Tuple (action int, hidden, attribution dict with mode semantics).
        """
        fused, eager_norm, revisit_norm, hidden, obs_1d = self._compute_fused(obs, eager_hidden)
        action = int(np.argmax(fused))
        attribution = self._attribution_for(action, obs_1d)
        # Legacy keys preserved for API consumers.
        eager_contrib = float(self.eager_weight * eager_norm[action])
        revisit_contrib = float(self.revisit_weight * revisit_norm[band_of_action(action, self.n_modes)])
        total = eager_contrib + revisit_contrib + 1e-8
        attribution["eager_pct"] = float(eager_contrib / total)
        attribution["revisit_pct"] = float(revisit_contrib / total)
        # Fused action score underlying the selection (action_score log field).
        attribution["action_score"] = float(fused[action])
        attribution["action"] = action
        logger.debug("MoE selected action=%d %s", action, attribution)
        return action, hidden, attribution

    def select_bands(
        self, obs: np.ndarray | torch.Tensor, eager_hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        k: int | None = None, return_full: bool = True,
    ) -> tuple[list[int], tuple[torch.Tensor, torch.Tensor] | None, dict[str, float]]:
        """Select top-K actions (or bands) via fused time-frequency scores.

        Args:
            obs: Observation vector (obs_dim,) numpy or (B,T,obs_dim) torch.
            eager_hidden: Optional LSTM hidden state (if torch obs).
            k: Number of top actions to return; defaults to k_receivers.
            return_full: If True, returns top-K *action* indices (time-frequency).
                If False, returns top-K *band* indices (decoded) for API customers.

        Returns:
            Tuple (selected_indices List[int] len K, hidden, attribution_dict
            {eager_pct, revisit_pct}).
        """
        fused, eager_norm, revisit_norm, hidden, obs_1d = self._compute_fused(obs, eager_hidden)

        # Top-K actions
        k_eff = self.k_receivers if k is None else int(k)
        k_eff = min(k_eff, self.n_actions)
        top_k = np.argsort(fused)[-k_eff:][::-1].tolist()

        # Attribution for explainability (computed over the top actions before decoding)
        eager_contrib = float(np.sum(self.eager_weight * eager_norm[top_k]))
        revisit_contrib = float(np.sum(
            self.revisit_weight * revisit_norm[[band_of_action(a, self.n_modes) for a in top_k]]
        ))
        total = eager_contrib + revisit_contrib + 1e-8
        attribution = {"eager_pct": float(eager_contrib / total), "revisit_pct": float(revisit_contrib / total)}

        if not return_full:
            # Decode to unique band indices (per action), dedup preserving order.
            bands = []
            for a in top_k:
                b = band_of_action(a, self.n_modes)
                if b not in bands:
                    bands.append(b)
            top_k = bands

        logger.debug("MoE fused top=%s attribution=%s", top_k, attribution)
        return top_k, hidden, attribution

    def update(self, selected_action: int) -> None:
        """Update revisit agent after a time-frequency action.

        Args:
            selected_action: Flat action index (band*n_modes + mode).
        """
        band = band_of_action(int(selected_action), self.n_modes)
        self.revisit_agent.update(band)

    def reset(self) -> None:
        """Reset both agents' episodic state and preemptive map."""
        self.eager_agent.reset()
        self.revisit_agent.reset()
        self._preemptive_urgency.fill(0.0)

    # Torch forward for batched training (keeps old API)
    def forward(
        self, obs: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]:
        """Batched fused scores for training.

        Args:
            obs: (B,T,obs_dim).
            hidden: LSTM hidden.

        Returns:
            Tuple (fused_scores (B,T,n_actions), next_hidden, attribution dict tensors).
        """
        q_values, _aux, next_hidden = self.drqn(obs, hidden)
        q_min = q_values.min(dim=-1, keepdim=True)[0]
        q_max = q_values.max(dim=-1, keepdim=True)[0]
        q_norm = (q_values - q_min) / (q_max - q_min + 1e-8)
        # Revisit urgency from the 10-feature observation layout — feature index 4
        # is the per-band normalized revisit age.
        n = self.n_bands
        features_per_band = obs.shape[-1] // n
        age_idx = 4  # normalized revisit age within each band's feature block
        if features_per_band > age_idx:
            b = torch.arange(n, device=obs.device) * features_per_band + age_idx
            time_since = obs[:, :, b]
        else:
            time_since = obs[:, :, :n]
        urgency = 1.0 - torch.exp(-self.decay_rate * time_since * 100.0)
        # Broadcast revisit urgency across modes: (B,T,n) -> (B,T,n*m).
        urgency_action = urgency.repeat(1, 1, self.n_modes)
        eager_contrib = self.eager_weight * q_norm
        revisit_contrib = self.revisit_weight * urgency_action
        fused = eager_contrib + revisit_contrib

        # Mode-semantic intent (Phase 5): per-(band, mode) reason scores so each
        # dwell mode is linked to its observable driver (revisit age, uncertainty,
        # periodic-imminent-arrival urgency) rather than being only a dwell length.
        n = self.n_bands
        if features_per_band > UNCERTAINTY_IDX:
            unc = obs[:, :, UNCERTAINTY_IDX :: features_per_band]
            rev = obs[:, :, REVISIT_AGE_IDX :: features_per_band]
            per = (
                torch.from_numpy(np.asarray(self._preemptive_urgency, dtype=np.float32))
                .to(obs.device)
                .view(1, 1, n)
                .expand(obs.shape[0], obs.shape[1], n)
            )
            peak = torch.maximum(rev, per)
            sem_per_band = torch.stack([
                0.30 * (1.0 - peak),                     # SHORT_DWELL (recce)
                torch.full_like(peak, 0.45),             # NORMAL_DWELL (surveillance)
                0.25 + 0.35 * unc,                       # LONG_DWELL (deep observation)
                0.20 + 0.60 * rev,                       # REVISIT
                0.20 + 0.60 * per,                       # PREEMPTIVE_INTERCEPT
            ], dim=-1)                                    # (B,T,n,5)
            sem_action = sem_per_band.reshape(obs.shape[0], obs.shape[1], n * self.n_modes)
            fused = fused + self.semantic_weight * sem_action

        attribution = {"eager_contribution": eager_contrib.detach(), "revisit_contribution": revisit_contrib.detach(), "raw_q_values": q_values.detach()}
        return fused, next_hidden, attribution

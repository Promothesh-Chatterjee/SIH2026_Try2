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

from ..contracts import (
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
from ..cognitive.temporal_predictor import TemporalPredictor
from ..cognitive.spatial_tracker import SpatialTracker

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

        def get_q(
            self, obs: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor] | None = None
        ) -> tuple[np.ndarray, tuple[torch.Tensor, torch.Tensor] | None]:
            """Run DRQN forward, return Q-values and next hidden.

            Args:
                obs: (B,T,obs_dim) or (obs_dim,) tensor.
                hidden: Optional (h, c) LSTM hidden tuple override.

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
            hx = hidden if hidden is not None else self.hidden
            with torch.inference_mode():
                q, _aux, h = self.drqn(obs_b, hx)
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

        step = update

        def scores(self) -> np.ndarray:
            """Return per-band urgency vector exp(decay * (t - last_visit)), normalised to [0,1].

            Returns:
                (n_bands,) float32 in [0,1].
            """
            raw = np.exp(self.decay_rate * (self.current_t - self.last_visit_time))
            # Min-max to [0,1] for fusion
            r_min, r_max = float(np.min(raw)), float(np.max(raw))
            if r_max - r_min < 1e-8:
                norm = np.zeros_like(raw, dtype=np.float32)
            else:
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
        self.semantic_weight: float = float(config.get("semantic_weight", 0.1))
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
        self.confidence_margin_threshold: float = float(config.get("confidence_margin_threshold", 0.020))
        self.default_tau: float = float(config.get("tau", 0.0))
        self._consecutive_empty_band: int = 0
        self._consecutive_empty_total: int = 0
        self._last_band: int = -1
        self._historical_hits = np.zeros(self.n_bands, dtype=np.int32)

        # Stage 3: Deterministic Temporal Predictor for frequency agility & latency optimization
        self.alpha_dirichlet: float = float(config.get("alpha_dirichlet", 0.0))
        self.temporal_predictor = TemporalPredictor(
            n_bands=self.n_bands,
            n_modes=self.n_modes,
            base_dwell_us=float(config.get("base_dwell_us", 500.0)),
            alpha_dirichlet=self.alpha_dirichlet,
        )
        self.enable_t0: bool = bool(config.get("enable_t0", False))
        self.enable_t1: bool = bool(config.get("enable_t1", False))
        self.lambda_p: float = float(config.get("lambda_p", 1.0))
        self.lambda_t: float = float(config.get("lambda_t", 0.5))
        self.lambda_d: float = float(config.get("lambda_d", 0.1))
        self.lambda_a: float = float(config.get("lambda_a", 0.2))

        # Cognitive Exploration Guard
        self.enable_exploration_guard: bool = bool(config.get("enable_exploration_guard", False))
        self.exploration_guard_confidence: float = float(config.get("exploration_guard_confidence", 0.45))
        self.exploration_guard_eta_us: float = float(config.get("exploration_guard_eta_us", 500.0))

        # Spatial / AoA Intelligence Layer (enrichment per EmitterTrack)
        self.spatial_tracker = SpatialTracker(n_sectors=int(config.get("n_spatial_sectors", 12)))
        self.enable_spatial: bool = bool(config.get("enable_spatial", False))
        self.lambda_spatial: float = float(config.get("lambda_spatial", 0.25))

        self._simulated_clock_us: float = 0.0

        # Keep direct refs for torch MoE forward
        self.drqn = drqn_agent
        self._config = config
        logger.info(
            "SmartScanMoE eager=%.1f revisit=%.1f preemptive=%.1f K=%d actions=%d (T0=%s, T1=%s, Spatial=%s, Guard=%s, Alpha=%.2f)",
            self.eager_weight, self.revisit_weight, self.preemptive_weight,
            self.k_receivers, self.n_actions, self.enable_t0, self.enable_t1, self.enable_spatial,
            self.enable_exploration_guard, self.alpha_dirichlet,
        )

    def set_stage3_modes(
        self,
        enable_t0: bool = False,
        enable_t1: bool = False,
        lambda_p: float | None = None,
        lambda_t: float | None = None,
        lambda_d: float | None = None,
        lambda_a: float | None = None,
        enable_spatial: bool = False,
        lambda_spatial: float | None = None,
        alpha_dirichlet: float | None = None,
        enable_exploration_guard: bool | None = None,
        exploration_guard_confidence: float | None = None,
        exploration_guard_eta_us: float | None = None,
    ) -> None:
        """Dynamically configure Stage 3 prediction, exploration guard, and spatial modes."""
        self.enable_t0 = bool(enable_t0)
        self.enable_t1 = bool(enable_t1)
        self.enable_spatial = bool(enable_spatial)
        if lambda_p is not None:
            self.lambda_p = float(lambda_p)
        if lambda_t is not None:
            self.lambda_t = float(lambda_t)
        if lambda_d is not None:
            self.lambda_d = float(lambda_d)
        if lambda_a is not None:
            self.lambda_a = float(lambda_a)
        if lambda_spatial is not None:
            self.lambda_spatial = float(lambda_spatial)
        if alpha_dirichlet is not None:
            self.alpha_dirichlet = float(alpha_dirichlet)
            self.temporal_predictor.alpha_dirichlet = float(alpha_dirichlet)
        if enable_exploration_guard is not None:
            self.enable_exploration_guard = bool(enable_exploration_guard)
        if exploration_guard_confidence is not None:
            self.exploration_guard_confidence = float(exploration_guard_confidence)
        if exploration_guard_eta_us is not None:
            self.exploration_guard_eta_us = float(exploration_guard_eta_us)

    def reset(self) -> None:
        """Reset internal state of all components."""
        self.eager_agent.reset()
        self.revisit_agent.reset()
        self.temporal_predictor.reset()
        self._consecutive_empty_band = 0
        self._consecutive_empty_total = 0
        self._last_band = -1
        self._historical_hits.fill(0)
        self._preemptive_urgency.fill(0.0)
        self._simulated_clock_us = 0.0

    def update(self, action: int) -> None:
        """Update revisit tracking and simulated clock when an action is executed."""
        band = band_of_action(int(action), self.n_modes)
        mode = mode_of_action(int(action), self.n_modes)
        self.revisit_agent.update(band)
        dwell_mult = DEFAULT_DWELL_MULTIPLIERS[mode] if mode < len(DEFAULT_DWELL_MULTIPLIERS) else 1.0
        self._simulated_clock_us += 500.0 * dwell_mult

    def update_detections(self, detections: list, current_time: float | None = None) -> None:
        """Ingest detected pulses into temporal predictor and spatial tracker."""
        if current_time is not None and current_time > self._simulated_clock_us:
            self._simulated_clock_us = float(current_time)
        for d in detections:
            if isinstance(d, dict):
                t = float(d.get("toa_us", d.get("time_us", self._simulated_clock_us)))
                f = float(d.get("frequency_mhz", 0.0))
                b = int(min(self.n_bands - 1, max(0, int(f // 500.0))))
                eid = int(d.get("emitter_id", 0))
                aoa = d.get("aoa_deg", d.get("angle_deg"))
            else:
                t = float(getattr(d, "toa_us", getattr(d, "time_us", self._simulated_clock_us)))
                f = float(getattr(d, "frequency_mhz", 0.0))
                b = int(min(self.n_bands - 1, max(0, int(f // 500.0))))
                eid = int(getattr(d, "emitter_id", 0))
                aoa = getattr(d, "aoa_deg", getattr(d, "angle_deg", None))
            self.temporal_predictor.update_from_pulse(eid, t, f, b)
            if aoa is not None and np.isfinite(aoa):
                self.spatial_tracker.update_from_track(eid, float(aoa), t)

    def update_result(
        self, hit: bool, band: int, detections: list | None = None, current_time: float | None = None
    ) -> None:
        """Track hit results for consecutive unproductive dwells, historical hits, and pulse arrivals."""
        if hit:
            self._consecutive_empty_band = 0
            self._consecutive_empty_total = 0
            if 0 <= int(band) < self.n_bands:
                self._historical_hits[int(band)] += 1
        else:
            self._consecutive_empty_total += 1
            if self._last_band == band or self._last_band == -1:
                self._consecutive_empty_band += 1
            else:
                self._consecutive_empty_band = 1
        self._last_band = int(band)
        if detections:
            self.update_detections(detections, current_time=current_time)

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
        """5-tuple compatibility wrapper (kept for existing unit tests)."""
        fused, eager_norm, revisit_norm, hidden, obs_1d, _raw_q = self._compute_fused_full(obs, eager_hidden)
        return fused, eager_norm, revisit_norm, hidden, obs_1d

    def _compute_fused_full(
        self, obs: np.ndarray | torch.Tensor, eager_hidden: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[torch.Tensor, torch.Tensor] | None, np.ndarray, np.ndarray]:
        """Compute fused per-action scores for numpy or torch observations.

        Returns:
            Tuple (fused (n_actions,), eager_norm (n_actions,), revisit_norm (n_bands,),
            hidden, obs_1d (obs_dim,) float32, raw_q (n_actions,) float32).
        """
        if isinstance(obs, np.ndarray):
            obs_1d = np.asarray(obs).reshape(-1).astype(np.float32)
            obs_t = torch.from_numpy(obs_1d)
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0).unsqueeze(0)
            elif obs_t.dim() == 2:
                obs_t = obs_t.unsqueeze(1)
            q_raw, hidden = self.eager_agent.get_q(obs_t.squeeze(0) if obs_t.shape[0] == 1 else obs_t, hidden=eager_hidden)
            eager_norm = self.eager_agent.normalised_scores(q_raw)
            raw_q = np.asarray(q_raw, dtype=np.float32).reshape(-1)
        elif isinstance(obs, torch.Tensor):
            obs_1d = obs[0, -1].detach().cpu().numpy().reshape(-1)
            q, _aux, hidden = self.drqn(obs, eager_hidden)
            self.eager_agent.hidden = hidden
            q_last = q[0, -1].detach().cpu().numpy() if q.dim() == 3 else q.detach().cpu().numpy()
            if q_last.ndim > 1:
                q_last = q_last[0]
            eager_norm = self.eager_agent.normalised_scores(q_last)
            raw_q = np.asarray(q_last, dtype=np.float32).reshape(-1)
        else:
            raise TypeError(f"Unsupported obs type {type(obs)}")

        revisit_norm = self.revisit_agent.scores()
        semantic = self._mode_semantic_scores(obs_1d)
        fused = self._fused_action_scores(eager_norm, revisit_norm, semantic)
        return fused, eager_norm, revisit_norm, hidden if "hidden" in locals() else eager_hidden, obs_1d, raw_q

    def _score_attribution(
        self, action: int, q_values: np.ndarray, eager_norm: np.ndarray,
        revisit_norm_per_band: np.ndarray, obs_1d: np.ndarray,
    ) -> dict[str, float | int]:
        """Per-decision numeric attribution of the fused score.

        The configured-weight * component terms sum to ``fused_score`` and the
        raw DRQN Q at the action is ``q_score``. ``drqn_rank`` is the 1-based
        rank of the selected action when sorting by raw Q descending (1 = the
        DRQN argmax); ``moe_rank`` is 1 because the fused score picked it.
        """
        revisit_action = np.repeat(revisit_norm_per_band, self.n_modes).astype(np.float32)
        preempt_action = np.repeat(self._preemptive_urgency, self.n_modes).astype(np.float32)
        semantic = self._mode_semantic_scores(obs_1d).reshape(-1).astype(np.float32)
        q_argmax_action = int(np.argmax(q_values))
        order = np.argsort(q_values)[::-1]
        drqn_rank = int(np.where(order == action)[0][0]) + 1 if action in order else None
        return {
            "q_score": float(q_values[action]),
            "eager_score": float(self.eager_weight * eager_norm[action]),
            "revisit_score": float(self.revisit_weight * revisit_action[action]),
            "semantic_score": float(self.semantic_weight * semantic[action]),
            "preemptive_score": float(self.preemptive_weight * preempt_action[action]),
            "fused_score": float(self.eager_weight * eager_norm[action]
                                + self.revisit_weight * revisit_action[action]
                                + self.preemptive_weight * preempt_action[action]
                                + self.semantic_weight * semantic[action]),
            "q_argmax_action": q_argmax_action,
            "q_argmax_band": band_of_action(q_argmax_action, self.n_modes),
            "q_argmax_mode": int(mode_of_action(q_argmax_action, self.n_modes)),
            "drqn_rank": drqn_rank,
            "moe_rank": 1,
            "same_argmax": bool(action == q_argmax_action),
        }

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
        self, obs: np.ndarray | torch.Tensor, eager_hidden: tuple[torch.Tensor, torch.Tensor] | None = None, tau: float | None = None
    ) -> tuple[int, tuple[torch.Tensor, torch.Tensor] | None, dict[str, float]]:
        """Select action via confidence gating: DRQN primary (stochastic), heuristics fallback."""
        eff_tau = self.default_tau if tau is None else float(tau)
        # 1. Compute components
        fused_scores, eager_norm, revisit_norm, hidden, obs_1d, q_values = self._compute_fused_full(obs, eager_hidden)

        # 2. Confidence evaluation (Background-relative Q-margin)
        band_max = np.array([np.max(q_values[b * self.n_modes : (b + 1) * self.n_modes]) for b in range(self.n_bands)])
        order_b = np.argsort(band_max)[::-1]
        q_top1 = float(band_max[order_b[0]])
        q_top2 = float(band_max[order_b[1]]) if len(order_b) > 1 else q_top1
        band_q_margin = q_top1 - q_top2
        q_median = float(np.median(band_max))
        c_bg = q_top1 - q_median

        fpb = max(1, int(obs_1d.size) // self.n_bands) if obs_1d is not None and obs_1d.size else 10
        occ_vec = np.clip(obs_1d[0::fpb][: self.n_bands], 0.0, 1.0) if fpb > 0 else np.zeros(self.n_bands, dtype=np.float32)
        unc_vec = np.clip(obs_1d[3::fpb][: self.n_bands], 0.0, 1.0) if fpb > 3 else np.zeros(self.n_bands, dtype=np.float32)
        age_vec = np.clip(obs_1d[4::fpb][: self.n_bands], 0.0, 1.0) if fpb > 4 else np.zeros(self.n_bands, dtype=np.float32)

        # 3. Decision Gate: DRQN confident when eager_weight > 0 and background-relative margin >= threshold
        is_confident = (self.eager_weight > 0.0) and (
            (c_bg >= self.confidence_margin_threshold) or (band_q_margin >= self.confidence_margin_threshold)
        )

        # 4. Multi-emitter candidate set
        per_vec = np.clip(self._preemptive_urgency[: self.n_bands], 0.0, 1.0)
        q_candidates = [int(b) for b in order_b if band_max[b] >= q_top1 - 0.05][:3]

        # Stage 3 Temporal Predictor candidates & arrival forecasts
        pred_candidates = []
        pred_probs = np.zeros(self.n_bands, dtype=np.float32)
        pred_etas = {}
        if self.enable_t0 or self.enable_t1:
            curr_t = self._simulated_clock_us
            pred_candidates = self.temporal_predictor.get_predicted_candidate_bands(current_time=curr_t, top_k=2)
            for p in self.temporal_predictor.predict_all(curr_t):
                if p.prediction_confidence > 0.2:
                    b_p = p.target_band
                    pred_probs[b_p] = max(pred_probs[b_p], float(p.prediction_confidence))
                    pred_etas[b_p] = min(pred_etas.get(b_p, float("inf")), float(p.eta_us))

        # Adaptive occupancy threshold: in dense stare require 0.15, in sparse or scan regimes adapt down to 0.02
        max_occ = float(np.max(occ_vec)) if len(occ_vec) > 0 else 0.0
        occ_threshold = max(0.02, min(0.15, 0.5 * max_occ))
        occ_candidates = [int(b) for b in range(self.n_bands) if occ_vec[b] >= occ_threshold][:3]
        per_candidates = [int(b) for b in range(self.n_bands) if per_vec[b] > 0.50][:2] if self.preemptive_weight > 0.0 else []
        all_candidates = list(dict.fromkeys(pred_candidates + per_candidates + q_candidates + occ_candidates))[:7]

        # Filter out repeatedly empty band to enforce escape
        if self._consecutive_empty_band >= 2 and self._last_band in all_candidates and len(all_candidates) > 1:
            all_candidates = [b for b in all_candidates if b != self._last_band]

        # Force exploration if 3 consecutive dwells anywhere produced zero hits
        force_exploration = bool(self._consecutive_empty_total >= 3)

        # Exploration Guard: protect high-confidence impending arrivals from being overridden
        has_guarded_arrival = False
        if force_exploration and self.enable_exploration_guard:
            curr_t = self._simulated_clock_us
            preds = self.temporal_predictor.predict_all(curr_t, horizon_us=self.exploration_guard_eta_us)
            for p in preds:
                if p.prediction_confidence >= self.exploration_guard_confidence and p.eta_us <= self.exploration_guard_eta_us:
                    has_guarded_arrival = True
                    break
            if has_guarded_arrival:
                force_exploration = False

        # Evaluate T1 Action-Conditioned Predictive Utility if enabled
        u_action = None
        if self.enable_t1 and not force_exploration and (self._consecutive_empty_band < 2):
            u_scores, u_telem = self.temporal_predictor.compute_action_conditioned_utility(
                q_values=q_values,
                current_time=self._simulated_clock_us,
                lambda_p=self.lambda_p,
                lambda_t=self.lambda_t,
                lambda_d=self.lambda_d,
                lambda_a=self.lambda_a,
            )
            pred_b_set = set(u_telem.get("predicted_bands", []))
            act_res = u_telem.get("actionable_reservation", None)
            if act_res is not None:
                res_b = int(act_res["target_band"])
                if res_b not in all_candidates:
                    all_candidates.insert(0, res_b)
                pred_b_set.add(res_b)

            if pred_b_set and all_candidates:
                # Rank candidates by U(b, m) + optional gated spatial priority
                cand_actions = [b * self.n_modes + m for b in all_candidates for m in range(self.n_modes)]
                if self.enable_spatial:
                    cand_act_scores = []
                    for a in cand_actions:
                        b = band_of_action(a, self.n_modes)
                        # Bounded spatial priority across tracks active in band b
                        s_prio = 0.0
                        for eid, track in getattr(self.temporal_predictor, "tracks", {}).items():
                            if getattr(track, "last_band", None) == b:
                                s_prio = max(s_prio, self.spatial_tracker.get_spatial_priority(eid, self._simulated_clock_us))
                        cand_act_scores.append(u_scores[a] + self.lambda_spatial * s_prio)
                    best_cand_act = int(cand_actions[int(np.argmax(cand_act_scores))])
                else:
                    best_cand_act = int(cand_actions[int(np.argmax([u_scores[a] for a in cand_actions]))])
                b_cand = band_of_action(best_cand_act, self.n_modes)
                if b_cand in pred_b_set or pred_probs[b_cand] >= 0.4:
                    u_action = best_cand_act

        mode = None
        if self.eager_weight == 0.0 and self.revisit_weight == 0.0 and self.semantic_weight > 0.0 and not force_exploration:
            action = int(np.argmax(fused_scores))
            best_b = band_of_action(action, self.n_modes)
            mode = int(mode_of_action(action, self.n_modes))
            reason = DWELL_MODE_SEMANTICS[mode]
        elif u_action is not None:
            best_b = band_of_action(u_action, self.n_modes)
            mode = int(mode_of_action(u_action, self.n_modes))
            act_res = u_telem.get("actionable_reservation", None) if "u_telem" in locals() else None
            if act_res is not None and best_b == act_res["target_band"]:
                reason = "Temporal_reservation_active"
                if hasattr(self.temporal_predictor, "reservation_manager"):
                    self.temporal_predictor.reservation_manager.mark_executed(act_res["reservation_id"])
            else:
                reason = "Predictive_utility_active"
        elif self.enable_t0 and pred_candidates and (pred_probs[pred_candidates[0]] >= 0.5) and not force_exploration and (self._consecutive_empty_band < 2):
            best_b = pred_candidates[0]
            reason = "Predictive_hop_intercept"
            eta = pred_etas.get(best_b, 500.0)
            if eta <= 125.0:
                mode = 0  # SHORT_DWELL
            elif eta <= 500.0:
                mode = 1  # NORMAL_DWELL
            else:
                mode = 2  # LONG_DWELL
        elif self.preemptive_weight > 0.0 and np.max(per_vec) > 0.5 and (self._consecutive_empty_band < 2):
            best_b = int(np.argmax(per_vec))
            reason = "Preemptive_intercept"
        elif is_confident and all_candidates and not force_exploration and (self._consecutive_empty_band < 2):
            cand_scores = np.array([
                self.eager_weight * (band_max[b] - q_median)
                + 0.5 * occ_vec[b]
                + 2.0 * self.preemptive_weight * per_vec[b]
                + (1.5 * pred_probs[b] if self.enable_t0 else 0.0)
                for b in all_candidates
            ])
            if self.enable_spatial:
                for idx_c, b in enumerate(all_candidates):
                    s_prio = 0.0
                    for eid, track in getattr(self.temporal_predictor, "tracks", {}).items():
                        if getattr(track, "last_band", None) == b:
                            s_prio = max(s_prio, self.spatial_tracker.get_spatial_priority(eid, self._simulated_clock_us))
                    cand_scores[idx_c] += self.lambda_spatial * s_prio
            if eff_tau > 0.0:
                probs = np.exp((cand_scores - np.max(cand_scores)) / max(1e-5, eff_tau))
                probs = probs / np.sum(probs)
                best_b = int(np.random.choice(all_candidates, p=probs))
            else:
                best_b = int(all_candidates[int(np.argmax(cand_scores))])
            reason = "DRQN_topk_active"
        elif occ_candidates and not force_exploration and (self._consecutive_empty_band < 2):
            # Cognitive fallback: occupied bands
            occ_vals = np.array([occ_vec[b] for b in occ_candidates])
            if eff_tau > 0.0:
                probs = np.exp(occ_vals / max(1e-5, eff_tau))
                probs = probs / np.sum(probs)
                best_b = int(np.random.choice(occ_candidates, p=probs))
            else:
                best_b = int(occ_candidates[int(np.argmax(occ_vals))])
            reason = "Occupancy_fallback"
        else:
            # Cognitive exploration: revisit + uncertainty + historical hits + preemptive urgency
            max_hist = float(np.max(self._historical_hits)) if len(self._historical_hits) > 0 else 0.0
            hist_norm = (self._historical_hits / max(1.0, max_hist)).astype(np.float32)
            explor_scores = 0.40 * revisit_norm + 0.35 * unc_vec + 0.15 * hist_norm + 0.10 * per_vec
            if self._consecutive_empty_band >= 1 and 0 <= self._last_band < self.n_bands:
                explor_scores[self._last_band] = -1e9
            best_b = int(np.argmax(explor_scores))
            reason = "Cognitive_exploration"

        if mode is None:
            # Mode hierarchy combining learned Q-values + uncertainty/safety
            occ = float(occ_vec[best_b])
            unc = float(unc_vec[best_b])
            age = float(age_vec[best_b])

            # If a single mode has a massive learned Q margin (e.g. unit test or specialized fine-tuning), honor it
            q_modes = q_values[best_b * self.n_modes : (best_b + 1) * self.n_modes]
            m_argmax = int(np.argmax(q_modes))
            q_norm = float(q_values[best_b * self.n_modes + 1])
            q_long = float(q_values[best_b * self.n_modes + 2])

            if float(q_modes[m_argmax] - np.partition(q_modes, -2)[-2] if len(q_modes) > 1 else 0.0) >= 1.0:
                mode = m_argmax
            elif unc > 0.6 or age > 0.6:
                mode = 2  # LONG_DWELL
            elif self._consecutive_empty_band >= 2 and occ < 0.1:
                mode = 0  # SHORT_DWELL
            elif q_long > q_norm + 0.05:
                mode = 2  # LONG_DWELL
            else:
                mode = 1  # NORMAL_DWELL

        action = best_b * self.n_modes + mode

        attribution = self._attribution_for(action, obs_1d)
        
        eager_contrib = float(self.eager_weight * eager_norm[action])
        revisit_contrib = float(self.revisit_weight * revisit_norm[band_of_action(action, self.n_modes)])
        total = eager_contrib + revisit_contrib + 1e-8
        
        attribution["eager_pct"] = float(eager_contrib / total)
        attribution["revisit_pct"] = float(revisit_contrib / total)
        attribution["action_score"] = float(fused_scores[action])
        attribution["action"] = action
        attribution["q_margin"] = float(band_q_margin)
        attribution["fallback_triggered"] = float(reason in ("Cognitive_exploration", "Occupancy_fallback"))
        attribution["exploration_mode_active"] = float(reason == "Cognitive_exploration")
        attribution["drqn_candidate_active"] = float(reason == "DRQN_topk_active")
        attribution["occupancy_candidate_active"] = float(reason == "Occupancy_fallback")
        attribution["preemptive_candidate_active"] = float(reason == "Preemptive_intercept")
        attribution["consecutive_empty_total"] = int(self._consecutive_empty_total)
        attribution["consecutive_empty_band"] = int(self._consecutive_empty_band)
        attribution["reason"] = reason

        # Phase 2 runtime decision telemetry fields
        raw_q_np = np.asarray(q_values, dtype=np.float32).reshape(-1)
        raw_drqn_action = int(np.argmax(raw_q_np))
        raw_drqn_band = int(band_of_action(raw_drqn_action, self.n_modes))
        raw_drqn_mode = int(mode_of_action(raw_drqn_action, self.n_modes))
        final_action = int(action)
        final_band = int(band_of_action(final_action, self.n_modes))
        final_mode = int(mode_of_action(final_action, self.n_modes))
        action_was_overridden = bool(final_action != raw_drqn_action)
        override_source = str(reason) if action_was_overridden else None
        exploration_source = "cognitive_exploration" if reason == "Cognitive_exploration" else ("force_exploration" if force_exploration else "none")
        q_selected = float(raw_q_np[final_action]) if 0 <= final_action < len(raw_q_np) else 0.0
        q_max = float(np.max(raw_q_np)) if len(raw_q_np) > 0 else 0.0
        q_mean = float(np.mean(raw_q_np)) if len(raw_q_np) > 0 else 0.0
        q_std = float(np.std(raw_q_np)) if len(raw_q_np) > 0 else 0.0

        attribution["raw_drqn_action"] = raw_drqn_action
        attribution["raw_drqn_band"] = raw_drqn_band
        attribution["raw_drqn_mode"] = raw_drqn_mode
        attribution["final_action"] = final_action
        attribution["final_band"] = final_band
        attribution["final_mode"] = final_mode
        attribution["action_was_overridden"] = action_was_overridden
        attribution["override_source"] = override_source
        attribution["exploration_source"] = exploration_source
        attribution["q_selected"] = q_selected
        attribution["q_max"] = q_max
        attribution["q_mean"] = q_mean
        attribution["q_std"] = q_std


        # Cognitive Decision Explanation fields for telemetry & dashboard
        attribution["guarded_arrival_active"] = float(has_guarded_arrival)
        attribution["dirichlet_alpha"] = float(self.alpha_dirichlet)

        best_p_obj = None
        for p in self.temporal_predictor.predict_all(self._simulated_clock_us):
            if p.target_band == best_b:
                if best_p_obj is None or p.prediction_confidence > best_p_obj.prediction_confidence:
                    best_p_obj = p
        
        if best_p_obj is not None:
            attribution["predicted_track_id"] = int(best_p_obj.track_id)
            attribution["predicted_band"] = int(best_p_obj.target_band)
            attribution["p_next_band"] = float(best_p_obj.band_probabilities[best_b])
            attribution["eta_us"] = float(best_p_obj.eta_us)
            attribution["agility_score"] = float(best_p_obj.agility_score)
            attribution["prediction_confidence"] = float(best_p_obj.prediction_confidence)
            s_track = self.spatial_tracker.tracks.get(best_p_obj.track_id, None)
            attribution["spatial_confidence"] = float(s_track.confidence) if s_track else 0.0
            attribution["aoa_deg"] = float(s_track.mean_aoa_deg) if s_track else -1.0
        else:
            attribution["predicted_track_id"] = -1
            attribution["predicted_band"] = -1
            attribution["p_next_band"] = 0.0
            attribution["eta_us"] = -1.0
            attribution["agility_score"] = 0.0
            attribution["prediction_confidence"] = 0.0
            attribution["spatial_confidence"] = 0.0
            attribution["aoa_deg"] = -1.0

        attribution["drqn_score"] = float(q_values[action])
        attribution["predictive_score"] = float(u_scores[action]) if (u_action is not None and "u_scores" in locals()) else 0.0
        s_prio = 0.0
        if self.enable_spatial and best_p_obj is not None:
            s_prio = self.spatial_tracker.get_spatial_priority(best_p_obj.track_id, self._simulated_clock_us)
        attribution["spatial_score"] = float(s_prio)
        attribution["exploration_pressure"] = float(self.revisit_weight * revisit_norm[best_b] + 0.35 * unc_vec[best_b])
        
        attribution.update(self._score_attribution(action, q_values, eager_norm, revisit_norm, obs_1d))
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
        fused, eager_norm, revisit_norm, hidden, obs_1d, q_values = self._compute_fused_full(obs, eager_hidden)

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

        # RC-2: raw-Q argmax vs fused top-k — did the MoE keep the DRQN pick?
        q_argmax_action = int(np.argmax(q_values))
        attribution["q_argmax_action"] = q_argmax_action
        attribution["q_argmax_band"] = band_of_action(q_argmax_action, self.n_modes)
        attribution["q_argmax_mode"] = int(mode_of_action(q_argmax_action, self.n_modes))
        attribution["same_argmax_fraction"] = float(1.0 if q_argmax_action in top_k else 0.0)

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
        urgency_action = urgency.repeat_interleave(self.n_modes, dim=-1)
        eager_contrib = self.eager_weight * q_norm
        revisit_contrib = self.revisit_weight * urgency_action
        fused = eager_contrib + revisit_contrib

        # Periodic preemptive pressure broadcast across modes
        preempt_per_band = (
            torch.from_numpy(np.asarray(self._preemptive_urgency, dtype=np.float32))
            .to(obs.device)
            .view(1, 1, n)
            .expand(obs.shape[0], obs.shape[1], n)
        )
        preempt_action = preempt_per_band.repeat_interleave(self.n_modes, dim=-1)
        fused = fused + self.preemptive_weight * preempt_action

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

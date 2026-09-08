"""
Unified 11-entry baseline suite for the SmartScan scheduler.

Every baseline operates on the canonical time-frequency action space
``action = band * n_modes + mode`` and exposes the same minimal interface:

  ``act(obs) -> (action, info)``, ``step(obs) -> int``, ``reset()``.

NN-based baselines additionally expose ``select_action(obs, hidden)`` and
``update(action)`` for MoE-style drivers, and ``set_periodic_urgency_vector``
where they consume periodic-imminent-arrival pressure.

Fairness contract: NO baseline receives ground truth. Every entry consumes only
the observation vector produced by the same cognitive env (plus, for the
periodic-aware variants, the belief's periodic urgency — itself derived purely
from observable intercept history by the env's interceptor). Drivers using
:func:`src.evaluation.baseline_suite_eval.run_baseline_episode` guarantee the
SAME receiver, RF world, dwell rules, episode duration, metrics, and action
space for all 11 entries.

The 11 entries (in documented evaluation order):
  1. sequential_sweep     — sweep bands 0..n-1 in strict ascending order
  2. round_robin          — cycle bands 0..n-1 (canonical order)
  3. random               — uniform over the full 0..n_actions-1 space
  4. fixed_periodic_scan  — fixed-period schedule: dwell band b for dwell_slots
  5. highest_occupancy    — argmax of occupancy feature
  6. highest_uncertainty  — argmax of uncertainty feature
  7. revisit_heuristic    — argmax of time-since-last-visit (quickest revisit)
  8. drqn                 — raw DRQN, greedy argmax over Q (no fusion)
  9. drqn_revisit         — DRQN Q fused with revisit urgency (no periodic)
 10. drqn_periodic        — DRQN Q fused with periodic intercept urgency
 11. full_moe             — SmartScanMoE: DRQN + revisit + periodic + semantics
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from src.contracts import (
    CANONICAL_N_BANDS,
    CANONICAL_N_MODES,
    DWELL_MODES,
    NORMAL_DWELL,
    band_of_action,
    mode_of_action,
    n_actions_for,
)
from .baseline_schedulers import _emitted_action, HighestOccupancyScheduler, HighestUncertaintyScheduler, RoundRobinScheduler
from .random_scheduler import RandomScheduler
from .drqn_scheduler import DRQNScheduler
from .smartscan_moe import SmartScanMoE

logger = logging.getLogger(__name__)

BASELINE_NAMES: tuple[str, ...] = (
    "sequential_sweep",
    "round_robin",
    "random",
    "fixed_periodic_scan",
    "highest_occupancy",
    "highest_uncertainty",
    "revisit_heuristic",
    "drqn",
    "drqn_revisit",
    "drqn_periodic",
    "full_moe",
)

EXTENDED_BASELINES: tuple[str, ...] = (
    "gate100k_ar",
    "t0_predictive_candidates",
    "t1_predictive_utility",
)

NN_BASELINES: frozenset[str] = frozenset({
    "drqn", "drqn_revisit", "drqn_periodic", "full_moe",
    "gate100k_ar", "t0_predictive_candidates", "t1_predictive_utility",
})

HEURISTIC_BASELINES: frozenset[str] = frozenset(set(BASELINE_NAMES) - NN_BASELINES)


def _mode_for(n_modes: int | None) -> int:
    """Neutral NORMAL_DWELL mode index for the given mode count."""
    return NORMAL_DWELL if n_modes else 0


class SequentialSweep:
    """Strict ascending band sweep, one NORMAL dwell per band per cycle.

    This is the phase-sweep primitive: no learning, no prioritisation — a
    straight frequency ramp over 0..n_bands-1.
    """

    def __init__(self, n_bands: int = CANONICAL_N_BANDS, n_modes: int | None = None) -> None:
        self.n_bands = int(n_bands)
        self.n_modes = n_modes
        self._t = 0

    def reset(self) -> None:
        self._t = 0

    def act(self, observation: Any) -> tuple[int, dict[str, Any]]:
        band = self._t % self.n_bands
        action = _emitted_action(self.n_modes, band)
        self._t += 1
        return action, {"source": "sequential_sweep", "band": band, "observation": observation}

    def step(self, observation: Any) -> int:
        return self.act(observation)[0]


class FixedPeriodicScan:
    """Fixed-period scan schedule: dwell band ``b`` for ``dwell_slots`` steps.

    The revisit cadence is a fixed constant — a band is re-dwelled exactly
    ``n_bands * dwell_slots`` steps after it was first scheduled. With
    ``dwell_slots=1`` this reduces to :class:`SequentialSweep`; larger slots
    emulate a fixed slot-dwell radar sweep.
    """

    def __init__(self, n_bands: int = CANONICAL_N_BANDS, n_modes: int | None = None, dwell_slots: int = 1) -> None:
        self.n_bands = int(n_bands)
        self.n_modes = n_modes
        self.dwell_slots = max(1, int(dwell_slots))
        self._t = 0

    def reset(self) -> None:
        self._t = 0

    def act(self, observation: Any) -> tuple[int, dict[str, Any]]:
        band = (self._t // self.dwell_slots) % self.n_bands
        action = _emitted_action(self.n_modes, band)
        self._t += 1
        return action, {"source": "fixed_periodic_scan", "band": band, "dwell_slots": self.dwell_slots, "observation": observation}

    def step(self, observation: Any) -> int:
        return self.act(observation)[0]


class RevisitHeuristic:
    """Dwell the most-overdue band: argmax of time since last visit.

    Purely algorithmic (no learning, no GT): each step the band with the largest
    gap since its last dwell wins, then its clock resets. Guarantees bounded
    revisit latency even in a fully non-stationary spectrum.
    """

    def __init__(self, n_bands: int = CANONICAL_N_BANDS, n_modes: int | None = None) -> None:
        self.n_bands = int(n_bands)
        self.n_modes = n_modes
        # Seed unvisited bands as if visited one slot ago, so an unvisited band
        # (age t+1) always outranks a just-visited one (age t) — no tie-lock.
        self.last_visit = np.full(n_bands, -1.0, dtype=np.float64)
        self._t = 0

    def reset(self) -> None:
        self._t = 0
        self.last_visit.fill(-1.0)

    def act(self, observation: Any) -> tuple[int, dict[str, Any]]:
        ages = self._t - self.last_visit
        band = int(np.argmax(ages))
        self.last_visit[band] = float(self._t)
        self._t += 1
        action = _emitted_action(self.n_modes, band)
        return action, {"source": "revisit_heuristic", "band": band, "age": float(ages[band]), "observation": observation}

    def step(self, observation: Any) -> int:
        return self.act(observation)[0]


class DRQNBaseline:
    """Raw greedy DRQN over the flat time-frequency action space.

    No fusion: ``action = argmax_a Q(s, a)``. Maintains its LSTM hidden state
    internally across ``act``/``step`` calls (episodic).
    """

    def __init__(self, drqn: DRQNScheduler, device: str = "cpu", tau: float = 0.15, mode_selection_policy: str = "band_first_decoupled") -> None:
        self.drqn = drqn
        self.device = device
        self.tau = float(tau)
        self.mode_selection_policy = mode_selection_policy
        self.n_bands = drqn.n_bands
        self.n_modes = drqn.n_modes
        self.n_actions = drqn.n_actions
        self.hidden: tuple[torch.Tensor, torch.Tensor] | None = None
        self.consecutive_empty: int = 0
        self.last_band: int = -1

    def reset(self) -> None:
        try:
            self.hidden = self.drqn.init_hidden(1, self.device)
        except Exception:
            self.hidden = None
        self.consecutive_empty = 0
        self.last_band = -1

    def act(self, observation: Any) -> tuple[int, dict[str, Any]]:
        obs_t = torch.from_numpy(np.asarray(observation, dtype=np.float32)).to(self.device)
        with torch.inference_mode():
            if obs_t.dim() == 1:
                obs_b = obs_t.unsqueeze(0).unsqueeze(0)
            elif obs_t.dim() == 2:
                obs_b = obs_t.unsqueeze(1)
            else:
                obs_b = obs_t
            q_out, _aux, self.hidden = self.drqn(obs_b, self.hidden)
            q_vals = q_out[0, -1].detach().cpu().numpy()

            if self.mode_selection_policy == "band_first_decoupled":
                obs_1d = obs_b[0, -1].detach().cpu().numpy().reshape(-1)
                band_max = [np.max(q_vals[b * self.n_modes : (b + 1) * self.n_modes]) for b in range(self.n_bands)]
                order_b = np.argsort(band_max)[::-1]
                q_top1 = float(band_max[order_b[0]])
                q_top2 = float(band_max[order_b[1]]) if len(order_b) > 1 else q_top1
                q_margin = q_top1 - q_top2

                if self.tau > 0.0:
                    top_k = min(3, len(order_b))
                    top_idx = order_b[:top_k]
                    top_q = np.array([band_max[b] for b in top_idx])
                    probs = np.exp((top_q - np.max(top_q)) / max(1e-5, self.tau))
                    probs = probs / np.sum(probs)
                    best_b = int(np.random.choice(top_idx, p=probs))
                else:
                    best_b = int(order_b[0])

                fpb = max(1, int(obs_1d.size) // self.n_bands) if obs_1d.size else 10
                occ = float(obs_1d[0::fpb][best_b]) if fpb > 0 else 0.0
                unc = float(obs_1d[3::fpb][best_b]) if fpb > 3 else 0.0
                age = float(obs_1d[4::fpb][best_b]) if fpb > 4 else 0.0

                q_norm = float(q_vals[best_b * self.n_modes + 1])
                q_long = float(q_vals[best_b * self.n_modes + 2])

                # Hierarchy:
                # 1. High uncertainty / stale track -> prefer LONG
                if unc > 0.6 or age > 0.6:
                    mode = 2  # LONG_DWELL
                # 2. Repeatedly unproductive on this band -> SHORT (quick look)
                elif self.consecutive_empty >= 2 and occ < 0.1:
                    mode = 0  # SHORT_DWELL
                # 3. Strong track + Q(LONG) significantly better than Q(NORMAL) -> LONG
                elif q_long > q_norm + 0.05:
                    mode = 2  # LONG_DWELL
                # 4. Default -> NORMAL
                else:
                    mode = 1  # NORMAL_DWELL

                action = best_b * self.n_modes + mode
            else:
                order = np.argsort(q_vals)[::-1]
                q_top1 = float(q_vals[order[0]])
                q_top2 = float(q_vals[order[1]]) if len(order) > 1 else q_top1
                q_margin = q_top1 - q_top2

                if self.tau > 0.0:
                    top_k = min(5, len(order))
                    top_idx = order[:top_k]
                    top_q = q_vals[top_idx]
                    probs = np.exp((top_q - np.max(top_q)) / max(1e-5, self.tau))
                    probs = probs / np.sum(probs)
                    action = int(np.random.choice(top_idx, p=probs))
                else:
                    action = int(order[0])

        return action, {
            "source": "drqn",
            "band": band_of_action(action, self.n_modes),
            "mode": mode_of_action(action, self.n_modes),
            "mode_name": DWELL_MODES[mode_of_action(action, self.n_modes)],
            "q_top1": q_top1,
            "q_top2": q_top2,
            "q_margin": q_margin,
            "raw_q": q_vals,
        }

    def step(self, observation: Any) -> int:
        return self.act(observation)[0]

    def select_action(
        self, obs: np.ndarray | torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[int, tuple[torch.Tensor, torch.Tensor] | None, dict[str, Any]]:
        if hidden is not None:
            self.hidden = hidden
        action, attr = self.act(obs)
        return action, self.hidden, attr

    def update_result(self, hit: bool, band: int) -> None:
        if not hit:
            if self.last_band == band or self.last_band == -1:
                self.consecutive_empty += 1
            else:
                self.consecutive_empty = 1
        else:
            self.consecutive_empty = 0
        self.last_band = band

    def update(self, action: int) -> None:
        pass  # pure DRQN has no internal revisit/periodic state


class MoEBaseline:
    """MoE-style fused baseline (DRQN+Revisit / DRQN+Periodic / full MoE).

    Wraps a :class:`SmartScanMoE` and mirrors the same interface the production
    eval harness expects (``select_action``, ``update``,
    ``set_periodic_urgency_vector``) plus the suite's uniform
    ``act``/``step``/``reset``.
    """

    def __init__(self, moe: SmartScanMoE, source: str) -> None:
        self.moe = moe
        self.source = source
        self.n_bands = moe.n_bands
        self.n_modes = moe.n_modes
        self.n_actions = moe.n_actions
        self.hidden: tuple[torch.Tensor, torch.Tensor] | None = None

    def reset(self) -> None:
        self.moe.reset()
        self.hidden = None

    def act(self, observation: Any) -> tuple[int, dict[str, Any]]:
        action, self.hidden, attr = self.moe.select_action(observation, self.hidden)
        attr = {"source": self.source, **attr}
        return action, attr

    def step(self, observation: Any) -> int:
        return self.act(observation)[0]

    def select_action(
        self, obs: np.ndarray | torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[int, tuple[torch.Tensor, torch.Tensor] | None, dict[str, Any]]:
        if hidden is not None:
            self.hidden = hidden
        action, attr = self.act(obs)
        return action, self.hidden, attr

    def update(self, action: int) -> None:
        self.moe.update(action)

    def update_result(self, hit: bool, band: int, detections: list | None = None, current_time: float | None = None) -> None:
        self.moe.update_result(hit, band, detections=detections, current_time=current_time)

    def update_detections(self, detections: list, current_time: float | None = None) -> None:
        if hasattr(self.moe, "update_detections"):
            self.moe.update_detections(detections, current_time)

    def set_stage3_modes(
        self,
        enable_t0: bool = False,
        enable_t1: bool = False,
        lambda_p: float | None = None,
        lambda_t: float | None = None,
        lambda_d: float | None = None,
        lambda_a: float | None = None,
    ) -> None:
        if hasattr(self.moe, "set_stage3_modes"):
            self.moe.set_stage3_modes(
                enable_t0=enable_t0,
                enable_t1=enable_t1,
                lambda_p=lambda_p,
                lambda_t=lambda_t,
                lambda_d=lambda_d,
                lambda_a=lambda_a,
            )

    def set_periodic_urgency_vector(self, urgency: np.ndarray | list | tuple) -> None:
        self.moe.set_periodic_urgency_vector(urgency)


def build_baseline(
    name: str,
    *,
    n_bands: int = CANONICAL_N_BANDS,
    n_modes: int | None = None,
    seed: int = 42,
    device: str = "cpu",
    drqn: DRQNScheduler | None = None,
    config: dict[str, Any] | None = None,
):
    """Build one of the 11 canonical baseline schedulers.

    Fairness: none of the returned schedulers ever touches ground truth; they
    see only the observation (and, for the periodic-aware variants, the belief's
    periodic-urgency vector fed by the driver — itself observable-history-only).

    Args:
        name: One of BASELINE_NAMES (case-insensitive).
        n_bands: Number of bands.
        n_modes: Dwell modes (None → canonical taxonomy).
        seed: RNG seed (random/DRQN init path).
        device: Torch device for the NN-based baselines.
        drqn: Optional pre-trained DRQNScheduler. If None and the baseline is
            NN-based, a freshly (randomly) initialised DRQN is built — wiring
            only, NOT a trained policy.
        config: Optional dict for MoE fusion ({eager_weight, revisit_weight,
            preemptive_weight, semantic_weight, decay_rate, k_receivers, ...}).

    Returns:
        A scheduler with ``act``/``step``/``reset`` (+ NN hooks for NN ones).

    Raises:
        ValueError: Unknown baseline name.
    """
    key = name.lower()
    if key not in BASELINE_NAMES and key not in EXTENDED_BASELINES:
        raise ValueError(
            f"Unknown baseline '{name}'. Expected one of {BASELINE_NAMES + EXTENDED_BASELINES}"
        )
    cfg = dict(config or {})
    resolved_modes = CANONICAL_N_MODES if n_modes is None else int(n_modes)
    if key == "sequential_sweep":
        return SequentialSweep(n_bands=n_bands, n_modes=resolved_modes)
    if key == "round_robin":
        return RoundRobinScheduler(n_bands=n_bands, n_modes=resolved_modes)
    if key == "random":
        return RandomScheduler(n_bands=n_bands, n_modes=resolved_modes, seed=seed)
    if key == "fixed_periodic_scan":
        return FixedPeriodicScan(n_bands=n_bands, n_modes=resolved_modes, dwell_slots=int(cfg.get("dwell_slots", 1)))
    if key == "highest_occupancy":
        return HighestOccupancyScheduler(n_bands=n_bands, n_modes=resolved_modes)
    if key == "highest_uncertainty":
        return HighestUncertaintyScheduler(n_bands=n_bands, n_modes=resolved_modes)
    if key == "revisit_heuristic":
        return RevisitHeuristic(n_bands=n_bands, n_modes=resolved_modes)

    # ── NN-based entries (8-11) ────────────────────────────────────────────────
    modes = resolved_modes
    n_actions = n_actions_for(n_bands, modes)
    # Derive obs_dim from the canonical band×feature contract unless overridden.
    features_per_band = int(cfg.get("features_per_band", 10))
    obs_dim = int(cfg.get("obs_dim", n_bands * features_per_band))
    if drqn is None:
        # Fresh random-init policy for wiring tests / cold-start suites.
        # Seed deterministically so identical seeds yield identical policies.
        torch.manual_seed(int(seed))
        drqn = DRQNScheduler(
            obs_dim=obs_dim,
            n_bands=n_bands,
            n_actions=n_actions,
            n_modes=modes,
            lstm_hidden=int(cfg.get("lstm_hidden", 64)),
            lstm_layers=int(cfg.get("lstm_layers", 1)),
        )

    if key == "drqn":
        drqn.eval()
        tau = float(cfg.get("tau", 0.15))
        return DRQNBaseline(drqn, device=device, tau=tau)

    # Fused variants: reuse SmartScanMoE with the fusion term toggled.
    moe_cfg: dict[str, Any] = {
        **cfg,
        "n_bands": n_bands,
        "n_modes": modes,
        "n_actions": n_actions,
        "device": device,
    }
    if key == "drqn_revisit":
        moe_cfg.setdefault("eager_weight", 0.6)
        moe_cfg.setdefault("revisit_weight", 0.4)
        moe_cfg["preemptive_weight"] = 0.0
        moe_cfg["semantic_weight"] = 0.0
        return MoEBaseline(SmartScanMoE(drqn, moe_cfg), source="drqn_revisit")
    if key == "drqn_periodic":
        moe_cfg.setdefault("eager_weight", 0.7)
        moe_cfg.setdefault("revisit_weight", 0.0)
        moe_cfg.setdefault("preemptive_weight", 0.3)
        moe_cfg["semantic_weight"] = 0.0
        return MoEBaseline(SmartScanMoE(drqn, moe_cfg), source="drqn_periodic")

    # full_moe / gate100k_ar: all fusion terms active (configurable, canonical defaults).
    moe_cfg.setdefault("eager_weight", 0.6)
    moe_cfg.setdefault("revisit_weight", 0.4)
    moe_cfg.setdefault("preemptive_weight", 0.3)
    moe_cfg.setdefault("semantic_weight", 1.0)

    if key in ("full_moe", "gate100k_ar"):
        moe_cfg["enable_t0"] = False
        moe_cfg["enable_t1"] = False
        return MoEBaseline(SmartScanMoE(drqn, moe_cfg), source=key)

    if key == "t0_predictive_candidates":
        moe_cfg["enable_t0"] = True
        moe_cfg["enable_t1"] = False
        return MoEBaseline(SmartScanMoE(drqn, moe_cfg), source="t0_predictive_candidates")

    if key == "t1_predictive_utility":
        moe_cfg["enable_t0"] = True
        moe_cfg["enable_t1"] = True
        return MoEBaseline(SmartScanMoE(drqn, moe_cfg), source="t1_predictive_utility")

    return MoEBaseline(SmartScanMoE(drqn, moe_cfg), source="full_moe")


# Backward-compatible aliases
DRQNRevisitBaseline = MoEBaseline
DRQNPeriodicBaseline = MoEBaseline
FullSmartScanMoE = MoEBaseline
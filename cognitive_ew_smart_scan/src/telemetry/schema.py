"""
RC-2 telemetry schema v2.0.

Shared, NaN-safe builders for the episode/validation telemetry records and the
metric helpers (Shannon entropy, reward reconstruction) used by both the
training loop and the RC-2 test-suite.

Conventions:
  * Undefined metrics are logged as JSON ``null``, never as fabricated zeros.
  * NaN / Inf floats are coerced to ``null`` by :func:`coerce` before JSON.
  * JSONL serialisation goes through :func:`coerce` then the RunManager writer.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

TELEMETRY_SCHEMA_VERSION = "2.0"

# Canonical per-episode core metrics (directive RC-2). Field names are the log
# keys; legacy names (pd, pfa, avg_reward, ep_reward, ep_hits, epsilon) are
# preserved so older readers keep working.
EPISODE_CORE_FIELDS = [
    "pd",
    "pfa",
    "intercept_rate",
    "avg_reward",
    "episode_reward",
    "ep_hits",
    "ep_steps",
    "coverage",
    "discovery_rate",
    "avg_intercept_time",
    "avg_intercept_time_error",
    "pct_correct",
    "selected_active",
    "spectrum_active",
]

# Canonical per-episode reward decomposition totals.
REWARD_COMPONENT_FIELDS = [
    "reward_novel",
    "reward_hit",
    "reward_miss",
    "reward_dwell_cost",
    "reward_false_alarm",
    "reward_timing",
    "reward_priority",
    "reward_info_gain",
    "reward_redundant",
    "reward_delay",
    "reward_staleness",
]

# Canonical per-eval core metrics.
VAL_CORE_FIELDS = [
    "val_pd",
    "val_pfa",
    "val_intercept_rate",
    "val_avg_reward",
    "val_reward",
    "val_hits",
    "val_steps",
    "val_coverage",
    "val_discovery_rate",
    "val_avg_intercept_time",
    "val_avg_intercept_time_error",
    "val_pct_correct",
    "val_selected_active",
    "val_spectrum_active",
    "val_n_scenarios",
]

# Canonical runtime decision telemetry fields (Phase 2 audit & decision traceability).
DECISION_TELEMETRY_FIELDS = [
    "raw_drqn_action",
    "raw_drqn_band",
    "raw_drqn_mode",
    "final_action",
    "final_band",
    "final_mode",
    "action_was_overridden",
    "override_source",
    "exploration_source",
    "q_selected",
    "q_max",
    "q_mean",
    "q_std",
]



def safe_float(value: Any) -> float | None:
    """Coerce a value to float, mapping NaN/Inf/None to ``None``."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def safe_int(value: Any) -> int | None:
    """Coerce a value to int, mapping NaN/Inf/None to ``None``."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def coerce(value: Any) -> Any:
    """Recursively make a telemetry record JSON-safe.

    numpy scalars/arrays and torch tensors are converted to plain numbers and
    lists; NaN/Inf become ``None`` (JSON null). Booleans stay booleans.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, np.generic):
        return coerce(value.item())
    if isinstance(value, np.ndarray):
        return [coerce(v) for v in np.asarray(value).reshape(-1).tolist()]
    if isinstance(value, dict):
        return {str(k): coerce(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [coerce(v) for v in value]
    if isinstance(value, float):
        return safe_float(value)
    if isinstance(value, int):
        return int(value)
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return coerce(value.item())
        except Exception:
            pass
    return value


def shannon_entropy(counts: Iterable[Any]) -> float | None:
    """Shannon entropy (bits) of a count distribution.

    Returns ``None`` for empty/zero-sum/non-finite count vectors (undefined,
    not a fabricated zero).
    """
    arr = np.asarray(list(counts), dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return None
    total = float(arr.sum())
    if not math.isfinite(total) or total <= 0.0:
        return None
    p = arr / total
    p = p[p > 0.0]
    if p.size == 0:
        return None
    return safe_float(-float((p * np.log2(p)).sum()))


def reward_reconstruction(
    reward_components: dict[str, Any],
    ep_reward: Any,
    tol_rel: float = 1e-3,
    tol_abs: float = 1.0,
) -> dict[str, Any]:
    """Rebuild the episode reward from component totals and validate.

    ``reward_components`` keys are the canonical REWARD_COMPONENT_FIELDS names.
    Returns positive/negative totals, the reconstructed total, the abs error
    versus the logged episode reward and an ok flag.

    Undefined (None/NaN) components are excluded from the totals; if *any*
    component is undefined the reconstruction is flagged as not-ok.
    """
    comps = {k: safe_float(reward_components.get(k)) for k in REWARD_COMPONENT_FIELDS}
    defined = {k: v for k, v in comps.items() if v is not None}
    missing = [k for k, v in comps.items() if v is None]
    positive = sum(v for v in defined.values() if v > 0.0)
    negative = sum(v for v in defined.values() if v < 0.0)
    total = sum(defined.values())
    ep_r = safe_float(ep_reward)
    if ep_r is not None:
        error = abs(total - ep_r)
        tol = max(tol_abs, tol_rel * abs(ep_r))
        ok = (error <= tol) and not missing
    else:
        error = None
        ok = False
    return {
        "reward_positive_total": safe_float(positive),
        "reward_negative_total": safe_float(negative),
        "reward_total_reconstructed": safe_float(total),
        "reward_reconstruction_error": safe_float(error),
        "reward_reconstruction_ok": bool(ok) if not missing else False,
    }


def make_episode_record(
    step: int,
    episode: int,
    core: dict[str, Any],
    reward_components: dict[str, Any],
    actions: dict[str, Any],
    learning: dict[str, Any],
    moe: dict[str, Any],
    band_priorities: Any = None,
    epsilon: Any = None,
    schema_version: str = TELEMETRY_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Build the full RC-2 episode telemetry record (JSON-safe already).

    Args:
        step: Global step at episode end.
        episode: Episode number.
        core: EPISODE_CORE_FIELDS values.
        reward_components: Component totals keyed by REWARD_COMPONENT_FIELDS.
        actions: unique/entropy/counts aggregates.
        learning: DRQN update aggregates.
        moe: MoE attribution aggregates.
        band_priorities: Optional legacy priority vector (occupancy features).
        epsilon: Exploration epsilon at episode end.
    """
    rec: dict[str, Any] = {
        "telemetry_schema_version": schema_version,
        "type": "episode",
        "step": safe_int(step),
        "episode": safe_int(episode),
    }
    for k in EPISODE_CORE_FIELDS:
        rec[k] = coerce(core.get(k))
    # Legacy aliases preserved for existing readers.
    rec["ep_reward"] = rec.get("episode_reward")
    rec.update(coerce(reward_components))
    rec.update(coerce(actions))
    rec.update(coerce(learning))
    rec.update(coerce(moe))
    rec.update(reward_reconstruction(reward_components, core.get("episode_reward")))
    rec["epsilon"] = safe_float(epsilon)
    if band_priorities is not None:
        rec["band_priorities"] = coerce(band_priorities)
    return coerce(rec)


def make_val_record(
    step: int,
    episode: int,
    core_val: dict[str, Any],
    reward_components: dict[str, Any],
    entropy: dict[str, Any],
    validation_set_id: Any = None,
    validation_files: Any = None,
    schema_version: str = TELEMETRY_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Build the full RC-2 validation telemetry record (JSON-safe already).

    ``entropy`` may carry ``val_action_entropy``, ``val_band_entropy``,
    ``val_mode_entropy`` plus averaged per-scenario counts if desired.
    """
    rec: dict[str, Any] = {
        "telemetry_schema_version": schema_version,
        "type": "val",
        "step": safe_int(step),
        "episode": safe_int(episode),
    }
    for k in VAL_CORE_FIELDS:
        rec[k] = coerce(core_val.get(k))
    for k in REWARD_COMPONENT_FIELDS:
        rec["val_" + k] = coerce(reward_components.get(k))
    rec.update(coerce(entropy))
    rec["validation_set_id"] = str(validation_set_id) if validation_set_id is not None else None
    if validation_files is not None:
        rec["validation_files"] = coerce(validation_files)
    # Legacy alias + prior schema: keep val_reward reachable both ways.
    rec["val_reward"] = rec.get("val_reward", rec.get("val_avg_reward"))
    return coerce(rec)


def make_decision_telemetry(
    *,
    raw_drqn_action: Any = None,
    raw_drqn_band: Any = None,
    raw_drqn_mode: Any = None,
    final_action: Any = None,
    final_band: Any = None,
    final_mode: Any = None,
    action_was_overridden: Any = None,
    override_source: Any = None,
    exploration_source: Any = None,
    q_selected: Any = None,
    q_max: Any = None,
    q_mean: Any = None,
    q_std: Any = None,
) -> dict[str, Any]:
    """Build a canonical runtime decision telemetry record for single-step traceability."""
    return coerce({
        "raw_drqn_action": safe_int(raw_drqn_action),
        "raw_drqn_band": safe_int(raw_drqn_band),
        "raw_drqn_mode": safe_int(raw_drqn_mode),
        "final_action": safe_int(final_action),
        "final_band": safe_int(final_band),
        "final_mode": safe_int(final_mode),
        "action_was_overridden": bool(action_was_overridden) if action_was_overridden is not None else None,
        "override_source": str(override_source) if override_source is not None else None,
        "exploration_source": str(exploration_source) if exploration_source is not None else None,
        "q_selected": safe_float(q_selected),
        "q_max": safe_float(q_max),
        "q_mean": safe_float(q_mean),
        "q_std": safe_float(q_std),
    })
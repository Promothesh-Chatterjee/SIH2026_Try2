"""Preflight checks that prevent invalid strict TSRD training runs.

Validates the canonical observation contract (36 bands x 10 features = 360) and
the canonical time-frequency action contract (n_modes dwell modes x n_bands =
n_actions) plus receiver and reward consistency.
"""

from __future__ import annotations

from pathlib import Path

from ..contracts import (
    CANONICAL_N_BANDS,
    CANONICAL_BAND_FEATURES,
    CANONICAL_OBS_DIM,
    CANONICAL_RECEIVER,
    DWELL_MODES,
    DEFAULT_DWELL_MULTIPLIERS,
    n_actions_for,
)
from ..data.tsrd_manifest import resolve_split_dirs
from ..utils.checkpoint_meta import FEATURE_ORDER

# Reward terms the scheduler must have configured before a strict run.
REQUIRED_REWARD_KEYS = (
    "w_hit",
    "w_novel",
    "w_miss",
    "w_timing",
    "w_priority",
    "w_information_gain",
    "w_false_alarm",
    "w_dwell_cost",
    "w_redundant_scan",
    "w_delay",
)


def validate_training_gate(
    *,
    data_root: str | Path,
    deinterleaver_checkpoint: str | Path,
    normalization_stats: str | Path,
    environment_config: dict,
    model_config: dict | None = None,
) -> list[str]:
    """Return blocking reasons for a strict real-TSRD scheduler run."""
    root = Path(data_root)
    errors: list[str] = []
    if not root.exists():
        errors.append(f"TSRD root does not exist: {root}")

    for mode in ("stare", "scan"):
        splits = resolve_split_dirs(root, mode)
        for split in ("train", "val"):
            directory = splits[split]
            if not directory.is_dir() or not any(directory.glob("*.h5")):
                errors.append(f"Missing {mode}/{split} .h5 files: {directory}")

    checkpoint = Path(deinterleaver_checkpoint)
    if not checkpoint.is_file():
        errors.append(f"Deinterleaver checkpoint missing: {checkpoint}")
    stats = Path(normalization_stats)
    if not stats.is_file():
        errors.append(f"Normalization statistics missing: {stats}")

    # --- Canonical observation contract ---
    n_bands = int(environment_config.get("n_bands", CANONICAL_N_BANDS))
    band_features = int(environment_config.get("band_features", CANONICAL_BAND_FEATURES))
    obs_dim = int(environment_config.get("obs_dim", n_bands * band_features))
    if n_bands != CANONICAL_N_BANDS or band_features != CANONICAL_BAND_FEATURES or obs_dim != CANONICAL_OBS_DIM:
        errors.append(
            f"Observation contract invalid: n_bands={n_bands}, "
            f"band_features={band_features}, obs_dim={obs_dim}"
        )

    # --- Canonical time-frequency action contract ---
    n_modes = int(environment_config.get("n_modes", len(DWELL_MODES)))
    n_actions = int(environment_config.get("n_actions", n_actions_for(n_bands, n_modes)))
    if n_modes != len(DWELL_MODES):
        errors.append(f"Action contract invalid: n_modes={n_modes} != {len(DWELL_MODES)}")
    if n_bands * n_modes != n_actions:
        errors.append(
            f"Action contract invalid: n_actions={n_actions} != n_bands*n_modes={n_bands * n_modes}"
        )

    # --- Receiver contract ---
    receiver_values = {
        "ibw_mhz": float(environment_config.get("ibw_mhz", 0.0)),
        "frequency_step_mhz": float(environment_config.get("frequency_step_mhz", 0.0)),
        "freq_min_mhz": float(environment_config.get("freq_min_mhz", 0.0)),
        "freq_max_mhz": float(environment_config.get("freq_max_mhz", 0.0)),
    }
    if receiver_values != CANONICAL_RECEIVER:
        errors.append(f"Receiver configuration invalid: {receiver_values}")

    # --- Reward contract ---
    reward_config = (model_config or {}).get("reward", {})
    missing = [k for k in REQUIRED_REWARD_KEYS if k not in reward_config]
    if missing:
        errors.append(f"Reward config missing terms: {missing}")

    return errors


def validate_dwell_contract(environment_config: dict, model_config: dict | None = None) -> list[str]:
    """Check the dwell-mode taxonomy and multipliers are consistent.

    The environment's base dwell must equal the model config's
    ``dwell_modes.base_dwell_time_us`` so the DRQN/MoE and env agree on timing.
    The configured mode multipliers must match the canonical taxonomy.
    """
    errors: list[str] = []
    if model_config:
        dwell_cfg = model_config.get("dwell_modes", {})
        base = float(dwell_cfg.get("base_dwell_time_us", 0.0))
        env_base = float(environment_config.get("dwell_time_us", 0.0))
        if base != env_base:
            errors.append(f"Dwell base mismatch: model_config={base} vs environment={env_base}")
        mode_list = dwell_cfg.get("mode_multipliers", [])
        if isinstance(mode_list, list) and mode_list:
            names = [str(m.get("name")) for m in mode_list]
            mults = tuple(float(m.get("multiplier", 1.0)) for m in mode_list)
            if names != list(DWELL_MODES):
                errors.append(f"Dwell mode names mismatch: {names}")
            if mults != DEFAULT_DWELL_MULTIPLIERS:
                errors.append(f"Dwell multipliers mismatch: {mults}")
    return errors


def require_training_gate(**kwargs: object) -> None:
    """Raise before training if any strict TSRD prerequisite is missing."""
    errors = validate_training_gate(**kwargs)  # type: ignore[arg-type]
    dwell_errors = validate_dwell_contract(  # type: ignore[arg-type]
        kwargs.get("environment_config", {}), kwargs.get("model_config")
    )
    if dwell_errors:
        errors.extend(dwell_errors)
    if errors:
        raise RuntimeError("NOT_READY: strict TSRD training gate failed:\n- " + "\n- ".join(errors))
"""Canonical contracts for the Cognitive EW SmartScan scheduler.

Single source of truth for the time-frequency action space:

* Dwell-mode taxonomy and their dwell-time multipliers.
* Action encoding/decoding: ``action = band * n_modes + mode_index``.
* ``n_actions = n_bands * n_modes``.

The observation contract (``n_bands=36``, ``band_features=10``, ``obs_dim=360``)
and the per-band 10-feature layout are defined below. Tests assert env/DRQN/MoE
all agree with these constants.
"""

from __future__ import annotations

from typing import Any
import numpy as np

CANONICAL_N_BANDS = 36
CANONICAL_BAND_FEATURES = 10
CANONICAL_OBS_DIM = CANONICAL_N_BANDS * CANONICAL_BAND_FEATURES

RF_FREQ_MIN_MHZ = 0.0
RF_FREQ_MAX_MHZ = 18_000.0
RF_IBW_MHZ = 500.0
RF_FREQUENCY_STEP_MHZ = 500.0
RF_BASE_DWELL_TIME_US = 500.0

# Band-major observation layout. Keep this order stable for checkpoints.
FEATURE_ORDER: tuple[str, ...] = (
    "occupancy",
    "det_rate",
    "miss_rate",
    "uncertainty",
    "revisit_age",
    "emitter_count",
    "deint_confidence",
    "pri_stability",
    "agility",
    "priority",
)

# Canonical dwell-mode taxonomy (order is the mode index — do not reorder).
DWELL_MODES: tuple[str, ...] = (
    "SHORT_DWELL",
    "NORMAL_DWELL",
    "LONG_DWELL",
    "REVISIT",
    "PREEMPTIVE_INTERCEPT",
)

# Multipliers applied to the base receiver dwell time (base_dwell_time_us * multiplier).
# These are the amplification/compression factors of each dwell strategy. REVISIT and
# PREEMPTIVE_INTERCEPT keep a neutral 1.0 multiplier: their distinct semantics come from
# behaviour (revisit sensitivity boost / intercept-window alignment), not dwell length.
DEFAULT_DWELL_MULTIPLIERS: tuple[float, ...] = (0.25, 1.0, 2.5, 1.0, 1.0)

# Named indices for readability.
SHORT_DWELL, NORMAL_DWELL, LONG_DWELL, REVISIT, PREEMPTIVE_INTERCEPT = range(len(DWELL_MODES))

DWELL_MODE_INDEX: dict[str, int] = {name: i for i, name in enumerate(DWELL_MODES)}

# Per-mode semantic intent, aligned with DWELL_MODES. The reason key is what the
# action-selection layer reports as the *driver* of a chosen mode (req: distinguish
# why a mode was selected).
DWELL_MODE_SEMANTICS: tuple[str, ...] = (
    "recce",                 # SHORT_DWELL        - fast reconnaissance
    "surveillance",          # NORMAL_DWELL       - neutral surveillance
    "deep_observation",      # LONG_DWELL         - deeper observation of an uncertain band
    "revisit",               # REVISIT            - prioritize a previously observed / overdue band
    "periodic_intercept",    # PREEMPTIVE_INTERCEPT - prioritize an imminent predicted intercept
)

# Canonical 10-feature band block indices (see FEATURE_ORDER above).
REVISIT_AGE_IDX = 4
UNCERTAINTY_IDX = 3
OCCUPANCY_IDX = 0

CANONICAL_N_MODES = len(DWELL_MODES)
CANONICAL_N_ACTIONS = CANONICAL_N_BANDS * CANONICAL_N_MODES


def n_modes() -> int:
    """Number of dwell modes (length of the canonical taxonomy)."""
    return len(DWELL_MODES)


def n_actions_for(n_bands: int, n_modes: int | None = None) -> int:
    """Joint time-frequency action count for a band count."""
    return int(n_bands) * int(n_modes if n_modes is not None else len(DWELL_MODES))


CANONICAL_RECEIVER: dict[str, float] = {
    "freq_min_mhz": RF_FREQ_MIN_MHZ,
    "freq_max_mhz": RF_FREQ_MAX_MHZ,
    "ibw_mhz": RF_IBW_MHZ,
    "frequency_step_mhz": RF_FREQUENCY_STEP_MHZ,
}


def validate_environment_config(config: dict | None) -> list[str]:
    """Canonical observation/action/receiver contract violations for an env config.

    Checks the canonical counts (n_bands=36, band_features=10, obs_dim=360,
    n_modes=5, n_actions=180), the RF receiver constants (band edges, IBW,
    frequency step) and internal consistency (obs_dim == n_bands*band_features
    and n_actions == n_bands*n_modes). Scale-downs such as ``n_bands=18`` used
    by fast tests are reported here; callers that must stay strictly canonical
    (API inference, strict TSRD training) react to every violation, while the
    env itself only enforces the internal shape consistency.
    """
    cfg = config or {}
    errors: list[str] = []

    n_bands = int(cfg.get("n_bands", CANONICAL_N_BANDS))
    band_features = int(cfg.get("band_features", CANONICAL_BAND_FEATURES))
    obs_dim = int(cfg.get("obs_dim", n_bands * band_features))
    n_modes = int(cfg.get("n_modes", CANONICAL_N_MODES))
    n_actions = int(cfg.get("n_actions", n_bands * n_modes))

    if n_bands != CANONICAL_N_BANDS:
        errors.append(f"n_bands={n_bands} != canonical {CANONICAL_N_BANDS}")
    if band_features != CANONICAL_BAND_FEATURES:
        errors.append(f"band_features={band_features} != canonical {CANONICAL_BAND_FEATURES}")
    if obs_dim != n_bands * band_features:
        errors.append(f"obs_dim={obs_dim} != n_bands*band_features ({n_bands * band_features})")
    if obs_dim != CANONICAL_OBS_DIM:
        errors.append(f"obs_dim={obs_dim} != canonical {CANONICAL_OBS_DIM}")
    if n_modes != CANONICAL_N_MODES:
        errors.append(f"n_modes={n_modes} != canonical {CANONICAL_N_MODES}")
    if n_actions != n_bands * n_modes:
        errors.append(f"n_actions={n_actions} != n_bands*n_modes ({n_bands * n_modes})")
    if n_actions != CANONICAL_N_ACTIONS:
        errors.append(f"n_actions={n_actions} != canonical {CANONICAL_N_ACTIONS}")

    for key, expected in CANONICAL_RECEIVER.items():
        value = float(cfg.get(key, -1.0))  # absent -> never matches -> reported
        if value != expected:
            errors.append(f"{key}={value} != canonical {expected}")

    return errors


def require_environment_config(config: dict | None) -> None:
    """Raise ``ValueError`` listing every canonical-contract violation in a config."""
    errors = validate_environment_config(config)
    if errors:
        raise ValueError("Non-canonical environment config:\n- " + "\n- ".join(errors))


def validate_action(action: Any, n_actions: int | None = None) -> int:
    """Validate that action is a valid integer action index within [0, n_actions).

    Rejects booleans (subclasses of int in Python), non-integers,
    and out-of-range action indices.

    Args:
        action: Candidate action index.
        n_actions: Total valid actions (defaults to CANONICAL_N_ACTIONS = 180).

    Returns:
        The validated action as an int.

    Raises:
        TypeError: If action is a bool or non-integer.
        ValueError: If action is < 0 or >= n_actions.
    """
    total = CANONICAL_N_ACTIONS if n_actions is None else int(n_actions)
    if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
        raise TypeError(f"Action must be an integer, got {type(action).__name__}: {action!r}")
    act_int = int(action)
    if act_int < 0 or act_int >= total:
        raise ValueError(f"Action {act_int} out of range [0, {total})")
    return act_int


def encode_action(
    band: int,
    mode: int | None = None,
    n_modes: int | None = None,
    n_bands: int | None = None,
) -> int:
    """Encode a (band, mode) selection into a flat action index.

    Args:
        band: Band index in [0, n_bands).
        mode: Mode index in [0, n_modes). Defaults to NORMAL_DWELL.
        n_modes: Override for the number of modes (defaults to canonical).
        n_bands: Override for the number of bands (defaults to canonical).

    Returns:
        Flat action = band * n_modes + mode.
    """
    if isinstance(band, bool) or not isinstance(band, (int, np.integer)):
        raise TypeError(f"Band must be an integer, got {type(band).__name__}: {band!r}")
    nb = int(n_bands if n_bands is not None else CANONICAL_N_BANDS)
    b = int(band)
    if b < 0 or b >= nb:
        raise ValueError(f"Band {b} out of range [0, {nb})")

    m = int(n_modes if n_modes is not None else len(DWELL_MODES))
    if mode is not None and (isinstance(mode, bool) or not isinstance(mode, (int, np.integer))):
        raise TypeError(f"Mode must be an integer, got {type(mode).__name__}: {mode!r}")
    mode_val = NORMAL_DWELL if mode is None else int(mode)
    if mode_val < 0 or mode_val >= m:
        raise ValueError(f"Mode {mode_val} out of range [0, {m})")

    return b * m + mode_val


def band_of_action(action: int, n_modes: int | None = None, n_actions: int | None = None) -> int:
    """Decode a flat action into the selected band."""
    m = int(n_modes if n_modes is not None else len(DWELL_MODES))
    total = n_actions if n_actions is not None else (CANONICAL_N_BANDS * m)
    valid_act = validate_action(action, n_actions=total)
    return valid_act // m


def mode_of_action(action: int, n_modes: int | None = None, n_actions: int | None = None) -> int:
    """Decode a flat action into the selected dwell-mode index."""
    m = int(n_modes if n_modes is not None else len(DWELL_MODES))
    total = n_actions if n_actions is not None else (CANONICAL_N_BANDS * m)
    valid_act = validate_action(action, n_actions=total)
    return valid_act % m


def mode_name(mode: int) -> str:
    """Name of a dwell-mode index."""
    return DWELL_MODES[mode]


def dwell_us_for(
    base_dwell_time_us: float,
    mode: int,
    multipliers: tuple[float, ...] | None = None,
) -> float:
    """Dwell duration for a mode given the base receiver dwell time.

    Args:
        base_dwell_time_us: Base dwell duration (µs) set by receiver config.
        mode: Dwell-mode index.
        multipliers: Per-mode dwell multipliers (defaults to canonical).

    Returns:
        base_dwell_time_us * multiplier(mode).
    """
    mul = multipliers if multipliers is not None else DEFAULT_DWELL_MULTIPLIERS
    if mode < 0 or mode >= len(mul):
        raise ValueError(f"mode {mode} out of range [0, {len(mul)})")
    return float(base_dwell_time_us) * float(mul[mode])
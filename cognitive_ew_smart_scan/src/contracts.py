"""Canonical contracts for the Cognitive EW SmartScan scheduler.

Single source of truth for the time-frequency action space:

* Dwell-mode taxonomy and their dwell-time multipliers.
* Action encoding/decoding: ``action = band * n_modes + mode_index``.
* ``n_actions = n_bands * n_modes``.

The observation contract (``n_bands=36``, ``band_features=10``, ``obs_dim=360``)
and the per-band 10-feature layout are defined in ``src.utils.checkpoint_meta``
(``FEATURE_ORDER``). Tests assert env/DRQN/MoE all agree with these constants.
"""

from __future__ import annotations

# Canonical dwell-mode taxonomy (order is the mode index — do not reorder).
DWELL_MODES: tuple[str, ...] = (
    "SHORT_DWELL",
    "NORMAL_DWELL",
    "LONG_DWELL",
    "REVISIT",
    "PREEMPTIVE_INTERCEPT",
)

# Multipliers applied to the base receiver dwell time (base_dwell_time_us * multiplier).
# These are the amplification/compression factors of each dwell strategy.
DEFAULT_DWELL_MULTIPLIERS: tuple[float, ...] = (0.25, 1.0, 2.5, 1.0, 0.5)

# Named indices for readability.
SHORT_DWELL, NORMAL_DWELL, LONG_DWELL, REVISIT, PREEMPTIVE_INTERCEPT = range(len(DWELL_MODES))

DWELL_MODE_INDEX: dict[str, int] = {name: i for i, name in enumerate(DWELL_MODES)}


def n_modes() -> int:
    """Number of dwell modes (length of the canonical taxonomy)."""
    return len(DWELL_MODES)


def n_actions_for(n_bands: int, n_modes: int | None = None) -> int:
    """Joint time-frequency action count for a band count."""
    return int(n_bands) * int(n_modes if n_modes is not None else len(DWELL_MODES))


def encode_action(band: int, mode: int | None = None, n_modes: int | None = None) -> int:
    """Encode a (band, mode) selection into a flat action index.

    Args:
        band: Band index in [0, n_bands).
        mode: Mode index in [0, n_modes). Defaults to NORMAL_DWELL.
        n_modes: Override for the number of modes (defaults to canonical).

    Returns:
        Flat action = band * n_modes + mode.
    """
    m = int(n_modes if n_modes is not None else len(DWELL_MODES))
    mode = NORMAL_DWELL if mode is None else int(mode)
    if mode < 0 or mode >= m:
        raise ValueError(f"mode {mode} out of range [0, {m})")
    return int(band) * m + mode


def band_of_action(action: int, n_modes: int | None = None) -> int:
    """Decode a flat action into the selected band."""
    m = int(n_modes if n_modes is not None else len(DWELL_MODES))
    return int(action) // m


def mode_of_action(action: int, n_modes: int | None = None) -> int:
    """Decode a flat action into the selected dwell-mode index."""
    m = int(n_modes if n_modes is not None else len(DWELL_MODES))
    return int(action) % m


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
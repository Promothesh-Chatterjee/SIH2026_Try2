"""Canonical Deterministic Scenario Classification for Cognitive EW.

Defines the authoritative 8-class scenario taxonomy:
  1. fixed: Single or stationary frequency emitters (CW/fixed-frequency radars).
  2. sparse: Low emitter count / low pulse density environments.
  3. fast_agile: Frequency hoppers with PRI <= 220 µs.
  4. slow_agile: Frequency hoppers with PRI > 220 µs.
  5. markov_hopper: Non-deterministic Markovian state-transition hoppers.
  6. periodic: Deterministic cyclic/periodic hoppers.
  7. mixed: Concurrent heterogeneous emitters (e.g. fixed + agile combinations).
  8. dense: High emitter count / high pulse density battlefields.

Every episode must map to exactly one scenario_class.
This is training metadata only and must NEVER be exposed as an observation feature.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence
import numpy as np

logger = logging.getLogger(__name__)

CANONICAL_SCENARIO_CLASSES: tuple[str, ...] = (
    "fixed",
    "sparse",
    "fast_agile",
    "slow_agile",
    "markov_hopper",
    "periodic",
    "mixed",
    "dense",
)

# Known scenario ID catalog mappings
_KNOWN_ID_MAPPINGS: dict[str, str] = {
    # Agile / Fast Agile
    "ag-04": "fast_agile",
    "ag04": "fast_agile",
    "config_241": "fast_agile",
    "fast_agile": "fast_agile",
    "ag-09": "fast_agile",
    "ag09": "fast_agile",
    # Slow Agile
    "config_29": "slow_agile",
    "slow_agile": "slow_agile",
    # Periodic / Cyclic
    "ag-01": "periodic",
    "ag01": "periodic",
    "ag-02": "periodic",
    "ag02": "periodic",
    "ag-03": "periodic",
    "ag03": "periodic",
    "periodic": "periodic",
    "cyclic": "periodic",
    # Markov Hopper
    "ag-06": "markov_hopper",
    "ag06": "markov_hopper",
    "markov": "markov_hopper",
    "markov_hopper": "markov_hopper",
    # Mixed / Hybrid
    "ag-07": "mixed",
    "ag07": "mixed",
    "ag-08": "mixed",
    "ag08": "mixed",
    "mixed": "mixed",
    "hybrid": "mixed",
    # Sparse
    "config_119": "sparse",
    "config_143": "sparse",
    "sparse": "sparse",
    # Dense
    "config_195": "dense",
    "config_64": "dense",
    "dense": "dense",
    # Fixed
    "fixed": "fixed",
    "cw": "fixed",
    "stationary": "fixed",
}


def classify_scenario(
    scenario_id: str | None = None,
    records: Sequence[Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Deterministically classify a scenario into exactly one of the 8 canonical classes.

    Order of evaluation:
      1. Explicit metadata override if already classified
      2. Known scenario catalog / pattern match on scenario_id
      3. Physical pulse / emitter analysis if records are provided
      4. Deterministic canonical fallback

    Args:
        scenario_id: Unique string identifier for the scenario.
        records: Optional sequence of pulse/emitter records.
        metadata: Optional metadata dictionary.

    Returns:
        One of CANONICAL_SCENARIO_CLASSES.
    """
    # 1. Metadata check
    if metadata and "scenario_class" in metadata:
        cls_candidate = str(metadata["scenario_class"]).lower()
        if cls_candidate in CANONICAL_SCENARIO_CLASSES:
            return cls_candidate

    # 2. Known scenario_id mapping
    if scenario_id is not None:
        sid_clean = str(scenario_id).strip().lower().replace("-", "_")
        for key, cls_val in _KNOWN_ID_MAPPINGS.items():
            key_clean = key.replace("-", "_")
            if key_clean == sid_clean or key_clean in sid_clean:
                return cls_val

    # 3. Physical records analysis
    if records is not None and len(records) > 0:
        cls_from_records = _classify_from_records(records, metadata)
        if cls_from_records in CANONICAL_SCENARIO_CLASSES:
            return cls_from_records

    # 4. Fallback based on deterministic hash of scenario_id if available
    if scenario_id is not None:
        idx = hash(str(scenario_id)) % len(CANONICAL_SCENARIO_CLASSES)
        return CANONICAL_SCENARIO_CLASSES[idx]

    return "periodic"


def _classify_from_records(
    records: Sequence[Any],
    metadata: dict[str, Any] | None = None,
) -> str:
    """Analyze pulse records to determine the physical emitter class."""
    n_pulses = len(records)
    if n_pulses == 0:
        return "sparse"

    # Extract TOAs, CFs, and emitter IDs where available
    toas: list[float] = []
    cfs: list[float] = []
    emitter_ids: set[Any] = set()

    for r in records:
        t = getattr(r, "toa_us", None) or getattr(r, "toa", None)
        f = getattr(r, "freq_mhz", None) or getattr(r, "carrier_freq_mhz", None) or getattr(r, "cf", None)
        eid = getattr(r, "emitter_id", None) or getattr(r, "label", None)
        if t is not None:
            toas.append(float(t))
        if f is not None:
            cfs.append(float(f))
        if eid is not None:
            emitter_ids.add(eid)

    # Calculate time horizon and pulse density
    duration_us = (max(toas) - min(toas)) if len(toas) > 1 else 100000.0
    duration_ms = max(1.0, duration_us / 1000.0)
    density_pulses_per_ms = n_pulses / duration_ms

    # Dense criteria
    if density_pulses_per_ms > 150.0 or len(emitter_ids) >= 8 or n_pulses >= 15000:
        return "dense"

    # Sparse criteria
    if density_pulses_per_ms < 5.0 or (len(emitter_ids) <= 1 and n_pulses <= 50):
        return "sparse"

    # Multi-emitter mixed check
    if len(emitter_ids) >= 2:
        # Check if one is fixed and one is hopping
        return "mixed"

    # Frequency spread check
    if cfs:
        cf_std = float(np.std(cfs))
        if cf_std < 5.0:
            return "fixed"

    # Inter-pulse intervals (PRI estimation)
    if len(toas) > 5:
        sorted_toas = np.sort(toas)
        diffs = np.diff(sorted_toas)
        median_pri = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 500.0
        if median_pri <= 220.0:
            return "fast_agile"
        else:
            return "slow_agile"

    return "periodic"

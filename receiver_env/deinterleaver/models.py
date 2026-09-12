"""Canonical Data Contracts and Internal Models for Phase 4 PDW Deinterleaving."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class TrackStatus(Enum):
    """Lifecycle state of an emitter track."""
    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    TERMINATED = "TERMINATED"


@dataclass
class TrackHistory:
    """Bounded history buffers for recent pulse measurements."""
    max_history: int = 128
    recent_toas: deque[float] = field(default_factory=lambda: deque(maxlen=128))
    recent_pw_us: deque[float] = field(default_factory=lambda: deque(maxlen=128))
    recent_frequency_mhz: deque[float] = field(default_factory=lambda: deque(maxlen=128))

    def __post_init__(self) -> None:
        if self.recent_toas.maxlen != self.max_history:
            self.recent_toas = deque(self.recent_toas, maxlen=self.max_history)
            self.recent_pw_us = deque(self.recent_pw_us, maxlen=self.max_history)
            self.recent_frequency_mhz = deque(self.recent_frequency_mhz, maxlen=self.max_history)

    def append(self, toa_us: float, pw_us: float, frequency_mhz: float) -> None:
        self.recent_toas.append(toa_us)
        self.recent_pw_us.append(pw_us)
        self.recent_frequency_mhz.append(frequency_mhz)

    def clear(self) -> None:
        self.recent_toas.clear()
        self.recent_pw_us.clear()
        self.recent_frequency_mhz.clear()


@dataclass
class EmitterStatistics:
    """Internal stability and dispersion statistics for an emitter track."""
    frequency_std_mhz: float = 0.0
    pw_std_us: float = 0.0
    pri_std_us: float = 0.0


@dataclass
class EmitterTrack:
    """Emitter Track representation emitted by Phase 4."""
    track_id: int
    emitter_id: str
    pulse_count: int
    mean_frequency_mhz: float
    mean_pw_us: float
    estimated_pri_us: float
    first_toa_us: float
    last_toa_us: float
    track_confidence: float

    # Internal lifecycle and diagnostic attributes
    status: TrackStatus = TrackStatus.TENTATIVE
    pri_confidence: float = 0.0
    pri_jitter_pct: float = 0.0
    history: TrackHistory = field(default_factory=TrackHistory)
    stats: EmitterStatistics = field(default_factory=EmitterStatistics)
    assigned_pdw_ids: Set[int] = field(default_factory=set)

    # Frequency agility & hopping diagnostic attributes
    frequency_history: deque[float] = field(default_factory=lambda: deque(maxlen=128))
    latest_frequency_mhz: float = 0.0
    frequency_span_mhz: float = 0.0
    frequency_hopping_detected: bool = False
    hop_rate_hz: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": int(self.track_id),
            "emitter_id": str(self.emitter_id),
            "pulse_count": int(self.pulse_count),
            "mean_frequency_mhz": round(float(self.mean_frequency_mhz), 2),
            "latest_frequency_mhz": round(float(self.latest_frequency_mhz), 2),
            "frequency_span_mhz": round(float(self.frequency_span_mhz), 2),
            "frequency_hopping_detected": bool(self.frequency_hopping_detected),
            "hop_rate_hz": round(float(self.hop_rate_hz), 1),
            "mean_pw_us": round(float(self.mean_pw_us), 2),
            "estimated_pri_us": round(float(self.estimated_pri_us), 1) if self.estimated_pri_us > 0 else 0.0,
            "first_toa_us": round(float(self.first_toa_us), 2),
            "last_toa_us": round(float(self.last_toa_us), 2),
            "track_confidence": round(float(self.track_confidence), 2),
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


@dataclass(frozen=True)
class DeinterleaverConfig:
    """Hardened operational parameters for the Phase 4.1 PDW Deinterleaver."""
    # Frequency gating
    frequency_gate_mhz: float = 0.50            # Base frequency gate (+/- 0.5 MHz)
    min_frequency_gate_mhz: float = 0.15        # Tightest adaptive frequency gate (+/- 0.15 MHz)

    # Pulse width gating (Collision-Tolerant)
    pw_gate_pct: float = 0.45                  # Accommodate overlap stretching (+/- 45%)

    # Track lifecycle
    min_confirm_pulses: int = 3                # Pulses required to promote TENTATIVE -> CONFIRMED
    track_expiry_pri_mult: float = 6.0         # Expiry timeout multiple of estimated PRI
    track_expiry_timeout_us: float = 250_000.0  # Fallback absolute timeout if PRI unknown (250 ms)
    max_history_len: int = 128                 # Bounded history deque length

    # Track merge gates
    merge_freq_gate_mhz: float = 0.30          # Merge frequency tolerance (+/- 0.3 MHz)
    merge_pri_gate_pct: float = 0.08           # Merge PRI relative tolerance (8%)
    merge_pw_gate_pct: float = 0.30            # Merge PW relative tolerance (30%)
    merge_min_confidence: float = 0.70         # Minimum confidence for merging

    # Association scoring weights (Tuned Config A / Adaptive)
    weight_freq: float = 0.35                  # Frequency proximity weight
    weight_pri: float = 0.40                   # PRI timing consistency weight
    weight_pw: float = 0.15                    # Pulse width similarity weight
    weight_conf: float = 0.10                  # PDW measurement confidence weight
    association_threshold: float = 0.60        # Association acceptance threshold

    # PRI estimation parameters
    min_pri_us: float = 10.0                   # Minimum physical radar PRI (10 us -> 100 kHz PRF)
    max_pri_us: float = 50_000.0               # Maximum physical radar PRI (50 ms -> 20 Hz PRF)
    pri_bin_width_us: float = 0.5              # PRI histogram bin width
    harmonic_tolerance_pct: float = 0.08       # 8% tolerance for missing pulse harmonic detection

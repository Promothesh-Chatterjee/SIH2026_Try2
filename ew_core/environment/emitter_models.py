"""Emitter models for the RF Spectrum Simulation.

Follows the strict No-Prior-Intel contract:
Operational intelligence (scan periods, dwell times, patterns, hop sets, PRIs)
is strictly private to each emitter instance and never exposed via public properties.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np


class BaseEmitter(ABC):
    """Abstract base class for all RF emitters."""

    def __init__(self, emitter_id: int = 0) -> None:
        self._emitter_id = int(emitter_id)

    @property
    def emitter_id(self) -> int:
        """Sanctioned emitter identifier for ground-truth telemetry and hit attribution."""
        return self._emitter_id

    @abstractmethod
    def step(self, t: int) -> bool:
        """Returns True if the emitter is actively transmitting at discrete time slot t."""
        pass

    @abstractmethod
    def get_band(self, t: int) -> int:
        """Returns the band index illuminated/transmitted on at discrete time slot t."""
        pass


class StaticEmitter(BaseEmitter):
    """Static continuous or pulsed emitter on a single fixed frequency band."""

    def __init__(
        self,
        band_idx: int,
        duty_cycle: float = 1.0,
        pri: int = 1,
        emitter_id: int = 1,
    ) -> None:
        super().__init__(emitter_id=emitter_id)
        self._band_idx = int(band_idx)
        self._duty_cycle = float(duty_cycle)
        self._pri = max(1, int(pri))

    def step(self, t: int) -> bool:
        """Determine if emitter is active during discrete time slot t."""
        if self._duty_cycle >= 1.0:
            return True
        if self._duty_cycle <= 0.0:
            return False
        slot_in_pri = t % self._pri
        active_slots = max(1, int(round(self._pri * self._duty_cycle)))
        return slot_in_pri < active_slots

    def get_band(self, t: int) -> int:
        """Return the fixed band index."""
        return self._band_idx


class FreqAgileEmitter(BaseEmitter):
    """Frequency-agile emitter that hops across a set of frequency bands."""

    def __init__(
        self,
        hop_set: List[int],
        hop_interval: int = 1,
        pattern: str = "fixed",
        duty_cycle: float = 1.0,
        pri: int = 1,
        emitter_id: int = 2,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(emitter_id=emitter_id)
        self._hop_set = [int(b) for b in hop_set]
        self._hop_interval = max(1, int(hop_interval))
        self._pattern = str(pattern).lower()
        self._duty_cycle = float(duty_cycle)
        self._pri = max(1, int(pri))
        self._seed = seed

    def step(self, t: int) -> bool:
        """Determine if emitter is active during discrete time slot t."""
        if self._duty_cycle >= 1.0:
            return True
        if self._duty_cycle <= 0.0:
            return False
        slot_in_pri = t % self._pri
        active_slots = max(1, int(round(self._pri * self._duty_cycle)))
        return slot_in_pri < active_slots

    def get_band(self, t: int) -> int:
        """Return the current hop band at time slot t."""
        hop_idx = t // self._hop_interval
        if self._pattern == "fixed":
            return self._hop_set[hop_idx % len(self._hop_set)]
        else:
            # Deterministic pseudo-random selection per hop index
            base_seed = 0 if self._seed is None else int(self._seed)
            rng = np.random.RandomState((base_seed + hop_idx * 10007) & 0xFFFFFFFF)
            choice_idx = int(rng.randint(0, len(self._hop_set)))
            return self._hop_set[choice_idx]


class PeriodicScanEmitter(BaseEmitter):
    """Periodic scanning radar emitter that sweeps through an azimuth/frequency pattern."""

    def __init__(
        self,
        scan_period: int,
        dwell_time: int,
        scan_pattern: List[int],
        emitter_id: int = 3,
    ) -> None:
        super().__init__(emitter_id=emitter_id)
        self._scan_period = max(1, int(scan_period))
        self._dwell_time = max(1, int(dwell_time))
        self._scan_pattern = [int(b) for b in scan_pattern]

    def step(self, t: int) -> bool:
        """Determine if the radar beam is illuminating the receiver during time slot t."""
        t_in_cycle = t % self._scan_period
        active_duration = len(self._scan_pattern) * self._dwell_time
        return t_in_cycle < active_duration

    def get_band(self, t: int) -> int:
        """Return the band illuminated during time slot t."""
        t_in_cycle = t % self._scan_period
        pattern_idx = min(len(self._scan_pattern) - 1, t_in_cycle // self._dwell_time)
        return self._scan_pattern[pattern_idx]

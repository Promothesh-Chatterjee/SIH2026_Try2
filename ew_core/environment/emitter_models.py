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


class RotatingBeamEmitter(BaseEmitter):
    """Models a radar/communication emitter with a mechanically rotating antenna beam.

    This is the canonical 'spatially scanning emitter' from the PS description.

    The emitter is only interceptable when its beam is pointed toward the receiver.
    This creates a time-gated intercept window: T_window = beam_width_deg / scan_rate_dps

    Key PS requirement: 'model should enable prediction of intercept time and
    interception ratio against spatially scanning emitters.'
    """

    def __init__(
        self,
        band_idx: int,
        freq_mhz: float,
        scan_rate_rpm: float = 6.0,         # rotations per minute (10 sec/scan)
        beam_width_deg: float = 5.0,         # antenna beam width in degrees
        initial_angle_deg: float = 0.0,      # starting beam angle
        receiver_angle_deg: float = 90.0,    # angle to our receiver (fixed geometry)
        power_dbm: float = -60.0,
        emitter_id: int = 4,
    ) -> None:
        super().__init__(emitter_id=emitter_id)
        self.band_idx = int(band_idx)
        self.freq_mhz = float(freq_mhz)
        self.scan_rate_dps = float(scan_rate_rpm * 360.0 / 60.0)  # degrees per second
        self.scan_period_us = float((360.0 / self.scan_rate_dps) * 1e6)  # µs per full rotation
        self.beam_width_deg = float(beam_width_deg)
        self.current_angle_deg = float(initial_angle_deg % 360.0)
        self.receiver_angle_deg = float(receiver_angle_deg % 360.0)
        self.power_dbm = float(power_dbm)

        # Intercept window duration: beam_width / scan_rate in microseconds
        self.intercept_window_us = float((beam_width_deg / self.scan_rate_dps) * 1e6)

    def update(self, elapsed_us: float) -> None:
        """Advance the beam angle by elapsed_us microseconds."""
        delta_deg = self.scan_rate_dps * elapsed_us / 1e6
        self.current_angle_deg = (self.current_angle_deg + delta_deg) % 360.0

    def is_interceptable(self) -> bool:
        """Return True if the beam is currently pointed within beam_width of receiver."""
        diff = abs(self.current_angle_deg - self.receiver_angle_deg) % 360.0
        if diff > 180.0:
            diff = 360.0 - diff
        return diff <= (self.beam_width_deg / 2.0)

    def time_to_next_intercept_us(self) -> float:
        """Compute how many microseconds until the beam next enters the intercept window.

        Returns 0.0 if currently interceptable.
        """
        if self.is_interceptable():
            return 0.0
        diff = (self.receiver_angle_deg - self.current_angle_deg) % 360.0
        half_bw = self.beam_width_deg / 2.0
        leading_edge_diff = (diff - half_bw) % 360.0
        return float(leading_edge_diff / self.scan_rate_dps * 1e6)

    def intercept_probability(self, dwell_time_us: float, prediction_error_us: float = 0.0) -> float:
        """Compute Pr(intercept) for a given dwell time at this band.

        P = min(1, dwell_time / intercept_window) * correction(prediction_error)
        Implements the PS formula: P_intercept = beam_dwell/scan_period * correction
        """
        if self.intercept_window_us <= 0:
            return 0.0
        base_p = min(1.0, float(dwell_time_us) / self.intercept_window_us)
        if prediction_error_us > 0 and self.intercept_window_us > 0:
            overlap = max(0.0, 1.0 - float(prediction_error_us) / self.intercept_window_us)
        else:
            overlap = 1.0
        return float(base_p * overlap)

    def step(self, t: int) -> bool:
        """BaseEmitter compatibility method based on discrete slot t."""
        return self.is_interceptable()

    def get_band(self, t: int) -> int:
        """Return the band index."""
        return self.band_idx

    def to_dict(self) -> dict:
        return {
            "type": "rotating_beam",
            "band_idx": self.band_idx,
            "freq_mhz": self.freq_mhz,
            "scan_rate_rpm": self.scan_rate_dps * 60.0 / 360.0,
            "beam_width_deg": self.beam_width_deg,
            "scan_period_us": self.scan_period_us,
            "intercept_window_us": self.intercept_window_us,
            "current_angle_deg": self.current_angle_deg,
            "interceptable": self.is_interceptable(),
        }

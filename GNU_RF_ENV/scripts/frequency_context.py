"""Frequency Context Bridge — Phase 3B.

Maps a GNU Radio LOCAL/BASEBAND frequency (kHz) into the EXISTING receiver's
LOGICAL RF frequency space (MHz):

    RF frequency estimate (MHz)
        =
    receiver center/reference frequency (MHz)
        +
    estimated local/baseband offset (kHz -> MHz)

This is a pure, testable component.  It has NO GNU Radio dependency and does
NOT import or duplicate the receiver.

Ground-truth policy
-------------------
The ONLY receiver-side input is the receiver's tuning / reference state
(center_frequency_mhz, and optionally ibw_mhz).  These are observable,
configured receiver state, NOT emitter ground truth.

The component NEVER uses:
- emitter_id
- true emitter RF frequency
- true emitter PW / PRI / AoA
- scenario metadata
- baseband frequency reinterpreted directly as logical RF (e.g. +100 kHz -> 0.1 MHz)

Units
-----
- center_frequency_mhz : float (receiver logical RF MHz; e.g. 3200.0 or 8000.0)
- local_frequency_khz  : float (GNU Radio baseband offset; e.g. +100.0 or -250.0)
- return frequency_mhz : float (= center + local/1000)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

__all__ = [
    "KHZ_TO_MHZ",
    "local_frequency_to_rf_frequency",
    "FrequencyContextError",
    "FrequencyContext",
]

# 1 kHz = 0.001 MHz  (exact decimal; see KHZ_TO_MHZ_DENOMINATOR)
KHZ_TO_MHZ = 1000.0  # 1000 kHz per MHz


class FrequencyContextError(ValueError):
    """Raised for invalid center frequency / context configuration."""


def local_frequency_to_rf_frequency(
    center_frequency_mhz: float,
    local_frequency_khz: float,
) -> float:
    """Map a GNU Radio local/baseband offset (kHz) to logical RF (MHz).

    RF(MHz) = center(MHz) + local(kHz) / 1000

    Parameters
    ----------
    center_frequency_mhz : float
        Receiver center / reference frequency in MHz.  Must be finite and > 0.
    local_frequency_khz : float
        Estimated local/baseband offset in kHz (may be negative / zero).

    Returns
    -------
    float
        Estimated logical RF frequency in MHz.
    """
    center = _validate_center(center_frequency_mhz)
    local = _validate_local(local_frequency_khz)
    return center + local / KHZ_TO_MHZ


def _validate_center(center_frequency_mhz: float) -> float:
    if not isinstance(center_frequency_mhz, (int, float)):
        raise FrequencyContextError(
            f"center_frequency_mhz must be numeric, got {center_frequency_mhz!r}"
        )
    c = float(center_frequency_mhz)
    if not math.isfinite(c):
        raise FrequencyContextError(
            f"center_frequency_mhz must be finite, got {center_frequency_mhz!r}"
        )
    if c <= 0:
        raise FrequencyContextError(
            f"center_frequency_mhz must be > 0, got {center_frequency_mhz!r}"
        )
    return c


def _validate_local(local_frequency_khz: float) -> float:
    if not isinstance(local_frequency_khz, (int, float)):
        raise FrequencyContextError(
            f"local_frequency_khz must be numeric, got {local_frequency_khz!r}"
        )
    f = float(local_frequency_khz)
    if not math.isfinite(f):
        raise FrequencyContextError(
            f"local_frequency_khz must be finite, got {local_frequency_khz!r}"
        )
    return f


class FrequencyContext:
    """Receiver tuning/reference context for local->RF mapping.

    Carries the receiver's observable center frequency (and optional
    instantaneous bandwidth) so the same context can be reused across many
    PDWs, and so window-consistency checks can be run against a real
    SieveReceiver.
    """

    def __init__(
        self,
        center_frequency_mhz: float,
        ibw_mhz: Optional[float] = None,
    ) -> None:
        self.center_frequency_mhz = _validate_center(center_frequency_mhz)
        if ibw_mhz is not None:
            if not isinstance(ibw_mhz, (int, float)):
                raise FrequencyContextError(f"ibw_mhz must be numeric, got {ibw_mhz!r}")
            ibw = float(ibw_mhz)
            if not math.isfinite(ibw) or ibw <= 0:
                raise FrequencyContextError(
                    f"ibw_mhz must be finite and > 0, got {ibw_mhz!r}"
                )
            self.ibw_mhz = ibw
        else:
            self.ibw_mhz = None

    def local_to_rf(self, local_frequency_khz: float) -> float:
        """Map a local/baseband offset (kHz) to logical RF (MHz)."""
        return local_frequency_to_rf_frequency(
            self.center_frequency_mhz,
            local_frequency_khz,
        )

    def rf_to_local(self, frequency_mhz: float) -> float:
        """Inverse: logical RF (MHz) -> local/baseband offset (kHz)."""
        if not isinstance(frequency_mhz, (int, float)):
            raise FrequencyContextError(f"frequency_mhz must be numeric, got {frequency_mhz!r}")
        f = float(frequency_mhz)
        if not math.isfinite(f):
            raise FrequencyContextError(f"frequency_mhz must be finite, got {frequency_mhz!r}")
        return (f - self.center_frequency_mhz) * KHZ_TO_MHZ

    def rf_window_mhz(self) -> Tuple[float, float]:
        """Receiver instantaneous window [center - ibw/2, center + ibw/2].

        Requires ibw_mhz to have been provided.
        """
        if self.ibw_mhz is None:
            raise FrequencyContextError(
                "ibw_mhz was not provided; cannot compute RF window"
            )
        lower = self.center_frequency_mhz - self.ibw_mhz / 2.0
        upper = self.center_frequency_mhz + self.ibw_mhz / 2.0
        return lower, upper

    def rf_in_window(self, frequency_mhz: float) -> bool:
        """True if the mapped RF frequency is inside the receiver's IBW window."""
        if self.ibw_mhz is None:
            raise FrequencyContextError(
                "ibw_mhz was not provided; cannot evaluate RF window"
            )
        if not math.isfinite(float(frequency_mhz)):
            return False
        lower, upper = self.rf_window_mhz()
        return lower <= float(frequency_mhz) <= upper

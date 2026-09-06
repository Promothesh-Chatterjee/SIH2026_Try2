"""Per-Tune RF Source Generator — Phase 3O.

Deterministic, numpy-only generator that produces complex IQ samples for a
SINGLE receiver tune.  The receiver's center frequency determines the
baseband/local representation of each emitter.

Key relationship
----------------
    local_frequency_khz = (rf_frequency_mhz - center_frequency_mhz) * 1000

The generator does NOT require emitter_id, scenario metadata, true RF, AoA,
or any ground-truth information.  The RF value is used ONLY to compute the
baseband offset relative to the supplied receiver center.

Design
------
One generator instance = one receiver tune = one center frequency.
Multiple emitters may be configured per tune, but each emitter's baseband
offset is derived from context, not from identity.

Units
-----
- center_frequency_mhz : float  (receiver tune, e.g. 3200.0)
- rf_frequency_mhz     : float  (logical emitter RF, e.g. 3200.1)
- local_frequency_khz  : float  (= (rf - center) * 1000)
- pulse_width_us       : float
- pri_us               : float
- amplitude            : float  (relative linear, NOT calibrated dBm)
- jitter_fraction      : float  (fractional PRI jitter, e.g. 0.01 = +/-1%)
- sample_rate          : float  (S/s, default 2e6)
- seed                 : int    (deterministic RNG)

Channel / noise model
---------------------
Follows the validated assumptions from live_rf_environment.py:
- Additive Gaussian noise with configurable amplitude
- Two-tap multipath channel (0.8 + 0.2j, delay 1 sample)
- Deterministic with the chosen seed

No ground truth is inserted into the observable PDW output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "PerTuneConfig",
    "EmitterConfig",
    "PerTuneGenerator",
]


# ================================================================
# Emitter configuration (one emitter within a tune)
# ================================================================


@dataclass
class EmitterConfig:
    """Configuration for a single emitter within a per-tune context.

    The baseband/local frequency is derived from the emitter's logical RF
    relative to the receiver center.  It is NOT set by the caller as an
    opaque offset.
    """

    rf_frequency_mhz: float
    pulse_width_us: float = 10.0
    pri_us: float = 100.0
    amplitude: float = 1.0
    jitter_fraction: float = 0.01
    seed: int = 42

    def baseband_frequency_khz(self, center_frequency_mhz: float) -> float:
        """Compute local/baseband offset from receiver center.

        Parameters
        ----------
        center_frequency_mhz : float
            Receiver center/tune frequency in MHz.

        Returns
        -------
        float
            Local/baseband frequency in kHz (= (rf - center) * 1000).
        """
        return (self.rf_frequency_mhz - center_frequency_mhz) * 1000.0


# ================================================================
# Per-tune configuration
# ================================================================


@dataclass
class PerTuneConfig:
    """Configuration for one receiver tune.

    One PerTuneConfig = one receiver center frequency = one tune dwell.
    """

    center_frequency_mhz: float
    emitters: List[EmitterConfig] = field(default_factory=list)
    sample_rate: float = 2e6
    noise_amplitude: float = 0.05
    noise_seed: int = 42
    channel_taps: Sequence[complex] = field(
        default_factory=lambda: [0.8 + 0j, 0.2 + 0j]
    )
    channel_seed: int = 123

    def validate(self) -> None:
        """Validate the configuration.  Raises ValueError on invalid input."""
        if not isinstance(self.center_frequency_mhz, (int, float)):
            raise ValueError(
                f"center_frequency_mhz must be numeric, "
                f"got {self.center_frequency_mhz!r}"
            )
        if not math.isfinite(self.center_frequency_mhz):
            raise ValueError(
                f"center_frequency_mhz must be finite, "
                f"got {self.center_frequency_mhz!r}"
            )
        if self.center_frequency_mhz <= 0:
            raise ValueError(
                f"center_frequency_mhz must be > 0, "
                f"got {self.center_frequency_mhz!r}"
            )
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0, got {self.sample_rate!r}")
        if self.noise_amplitude < 0:
            raise ValueError(
                f"noise_amplitude must be >= 0, got {self.noise_amplitude!r}"
            )


# ================================================================
# Per-Tune Generator
# ================================================================


class PerTuneGenerator:
    """Deterministic IQ generator for a single receiver tune.

    Usage::

        gen = PerTuneGenerator()
        gen.configure(center_frequency_mhz=3200.0)
        gen.add_emitter(EmitterConfig(rf_frequency_mhz=3200.1, ...))
        iq = gen.generate_iq(duration_us=500.0)

    The generator produces complex64 IQ samples at the receiver's sample rate.
    Each emitter's local/baseband offset is derived from:

        local_khz = (rf_mhz - center_mhz) * 1000

    No ground truth (emitter_id, scenario metadata) is included in the output.
    """

    def __init__(self) -> None:
        self._config: Optional[PerTuneConfig] = None
        self._emitter_rngs: Dict[int, np.random.Generator] = {}

    # ----------------------------------------------------------------
    # Configuration
    # ----------------------------------------------------------------

    def configure(self, center_frequency_mhz: float, **kwargs: Any) -> PerTuneConfig:
        """Configure the generator for a specific receiver tune.

        Parameters
        ----------
        center_frequency_mhz : float
            Receiver center/tune frequency in MHz.
        **kwargs
            Additional PerTuneConfig fields (sample_rate, noise_amplitude, etc.).

        Returns
        -------
        PerTuneConfig
            The created configuration.
        """
        emitters = kwargs.pop("emitters", [])
        config = PerTuneConfig(
            center_frequency_mhz=center_frequency_mhz,
            emitters=list(emitters),
            **kwargs,
        )
        config.validate()
        self._config = config
        self._emitter_rngs = {}
        for i, emitter in enumerate(config.emitters):
            self._emitter_rngs[i] = np.random.default_rng(emitter.seed)
        return config

    def add_emitter(self, emitter: EmitterConfig) -> None:
        """Add an emitter to the current tune configuration.

        The emitter's local/baseband frequency is computed from its RF
        relative to the configured receiver center.

        Parameters
        ----------
        emitter : EmitterConfig
            Emitter to add.  Must have valid RF frequency.
        """
        if self._config is None:
            raise RuntimeError("Call configure() before add_emitter()")
        if not isinstance(emitter, EmitterConfig):
            raise TypeError(
                f"emitter must be an EmitterConfig, got {type(emitter)!r}"
            )
        if not math.isfinite(emitter.rf_frequency_mhz):
            raise ValueError(
                f"emitter rf_frequency_mhz must be finite, "
                f"got {emitter.rf_frequency_mhz!r}"
            )
        if emitter.rf_frequency_mhz <= 0:
            raise ValueError(
                f"emitter rf_frequency_mhz must be > 0, "
                f"got {emitter.rf_frequency_mhz!r}"
            )
        if not math.isfinite(emitter.pulse_width_us):
            raise ValueError(
                f"emitter pulse_width_us must be finite, got {emitter.pulse_width_us!r}"
            )
        if emitter.pulse_width_us <= 0:
            raise ValueError(
                f"emitter pulse_width_us must be > 0, "
                f"got {emitter.pulse_width_us!r}"
            )
        if not math.isfinite(emitter.pri_us):
            raise ValueError(
                f"emitter pri_us must be finite, got {emitter.pri_us!r}"
            )
        if emitter.pri_us <= 0:
            raise ValueError(
                f"emitter pri_us must be > 0, got {emitter.pri_us!r}"
            )
        if emitter.pri_us <= emitter.pulse_width_us:
            raise ValueError(
                f"emitter pri_us ({emitter.pri_us}) must be > "
                f"pulse_width_us ({emitter.pulse_width_us})"
            )
        if not math.isfinite(emitter.jitter_fraction):
            raise ValueError(
                f"emitter jitter_fraction must be finite, got {emitter.jitter_fraction!r}"
            )
        if emitter.jitter_fraction < 0:
            raise ValueError(
                f"emitter jitter_fraction must be >= 0, "
                f"got {emitter.jitter_fraction!r}"
            )
        if not math.isfinite(emitter.amplitude):
            raise ValueError(
                f"emitter amplitude must be finite, got {emitter.amplitude!r}"
            )
        if emitter.amplitude < 0:
            raise ValueError(
                f"emitter amplitude must be >= 0, "
                f"got {emitter.amplitude!r}"
            )

        self._config.emitters.append(emitter)
        self._emitter_rngs[len(self._config.emitters) - 1] = (
            np.random.default_rng(emitter.seed)
        )

    @property
    def center_frequency_mhz(self) -> float:
        """Current receiver center/tune frequency in MHz."""
        if self._config is None:
            raise RuntimeError("Call configure() first")
        return self._config.center_frequency_mhz

    @property
    def sample_rate(self) -> float:
        """Current sample rate in S/s."""
        if self._config is None:
            raise RuntimeError("Call configure() first")
        return self._config.sample_rate

    # ----------------------------------------------------------------
    # IQ generation
    # ----------------------------------------------------------------

    def generate_iq(
        self,
        duration_us: float,
        include_noise: bool = True,
    ) -> np.ndarray:
        """Generate complex IQ samples for the configured tune.

        Parameters
        ----------
        duration_us : float
            Duration of the output in microseconds.
        include_noise : bool
            If True, add Gaussian noise and apply the two-tap multipath
            channel (matching the validated channel model from
            live_rf_environment.py).  Default True.

        Returns
        -------
        np.ndarray
            Complex64 IQ array of length = int(round(duration_us * sample_rate / 1e6)).
        """
        if self._config is None:
            raise RuntimeError("Call configure() before generate_iq()")

        config = self._config
        n_samples = max(1, int(round(duration_us * config.sample_rate / 1e6)))

        # Start with zeros.
        iq = np.zeros(n_samples, dtype=np.complex64)

        # Generate each emitter's contribution.
        for idx, emitter in enumerate(config.emitters):
            local_khz = emitter.baseband_frequency_khz(
                config.center_frequency_mhz
            )
            local_hz = local_khz * 1e3  # Convert kHz to Hz
            local_rad_per_sample = (
                2.0 * np.pi * local_hz / config.sample_rate
            )

            # Generate jittered pulse envelope.
            rng = self._emitter_rngs[idx]
            envelope = self._generate_pulse_envelope(
                n_samples=n_samples,
                pulse_width_us=emitter.pulse_width_us,
                pri_us=emitter.pri_us,
                jitter_fraction=emitter.jitter_fraction,
                amplitude=emitter.amplitude,
                rng=rng,
            )

            # Modulate onto carrier at local/baseband frequency.
            t_samples = np.arange(n_samples, dtype=np.float64)
            carrier = np.exp(
                1j * local_rad_per_sample * t_samples
            ).astype(np.complex64)

            emitter_iq = (envelope * carrier).astype(np.complex64)
            iq = iq + emitter_iq

        # Apply noise and channel if requested.
        if include_noise and config.noise_amplitude > 0:
            iq = self._apply_noise(iq, config)
            iq = self._apply_channel(iq, config)

        return iq.astype(np.complex64)

    # ----------------------------------------------------------------
    # Pulse envelope generation (reuses validated jitter model)
    # ----------------------------------------------------------------

    @staticmethod
    def _generate_pulse_envelope(
        n_samples: int,
        pulse_width_us: float,
        pri_us: float,
        jitter_fraction: float,
        amplitude: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate a pulse envelope with jittered PRI.

        Uses the same validated jitter model as live_rf_environment.py:
        uniform fractional PRI jitter within +/- jitter_fraction.

        Parameters
        ----------
        n_samples : int
            Total output samples.
        pulse_width_us : float
            Pulse width in microseconds.
        pri_us : float
            Nominal PRI in microseconds.
        jitter_fraction : float
            Fractional PRI jitter (0.0 = no jitter).
        amplitude : float
            Pulse amplitude.
        rng : np.random.Generator
            Seeded RNG for deterministic jitter.

        Returns
        -------
        np.ndarray
            Float64 envelope array (amplitude during pulse, 0 elsewhere).
        """
        envelope = np.zeros(n_samples, dtype=np.float64)

        # Pulse width in samples.
        pulse_samples = max(1, int(round(pulse_width_us * 2.0)))

        # Nominal PRI in samples.
        nominal_pri_samples = max(
            pulse_samples + 1,
            int(round(pri_us * 2.0)),
        )

        i = 0
        while i < n_samples:
            # Start a new pulse.
            end = min(i + pulse_samples, n_samples)
            envelope[i:end] = amplitude

            # Compute next PRI with jitter.
            if jitter_fraction > 0:
                jitter = rng.uniform(-jitter_fraction, jitter_fraction)
                actual_pri_s = pri_us * (1.0 + jitter)
                next_pri_samples = max(
                    pulse_samples + 1,
                    int(round(actual_pri_s * 2.0)),
                )
            else:
                next_pri_samples = nominal_pri_samples

            i = i + next_pri_samples

        return envelope

    # ----------------------------------------------------------------
    # Noise / channel
    # ----------------------------------------------------------------

    @staticmethod
    def _apply_noise(
        iq: np.ndarray,
        config: PerTuneConfig,
    ) -> np.ndarray:
        """Additive Gaussian noise (matches live_rf_environment.py)."""
        rng = np.random.default_rng(config.noise_seed)
        n = len(iq)
        noise = (
            config.noise_amplitude * rng.standard_normal(n)
            + 1j * config.noise_amplitude * rng.standard_normal(n)
        )
        return iq + noise.astype(np.complex64)

    @staticmethod
    def _apply_channel(
        iq: np.ndarray,
        config: PerTuneConfig,
    ) -> np.ndarray:
        """Two-tap multipath channel (matches live_rf_environment.py)."""
        if len(config.channel_taps) == 0:
            return iq

        taps = np.array(config.channel_taps, dtype=np.complex64)
        # Convolve: output = signal * taps (direct path + delayed path).
        out = np.convolve(iq, taps, mode="full")[: len(iq)]
        return out.astype(np.complex64)

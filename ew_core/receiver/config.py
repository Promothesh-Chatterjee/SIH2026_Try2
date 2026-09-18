"""Configuration dataclass for the Phase 1 EW Receiver Front End & Pulse Detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ReceiverConfig:
    """Configuration parameters for EW receiver front end and pulse detection.

    All components (filters, AGC, noise estimator, threshold detector,
    and pulse tracker) derive their runtime parameters strictly from this object.
    """

    # Sampling & RF front end
    sample_rate_hz: float = 20_000_000.0        # 20 MHz default sampling rate (50 ns/sample)
    center_frequency_hz: float = 3_000_000_000.0 # 3 GHz default tuner center
    bandwidth_hz: float = 10_000_000.0          # 10 MHz intermediate bandwidth
    filter_type: str = "bandpass"               # "bandpass" or "lowpass"
    filter_order: int = 4                       # IIR SOS filter order

    # Automatic Gain Control
    agc_enabled: bool = True
    agc_attack: float = 0.05                    # Fast attack coefficient
    agc_decay: float = 0.001                    # Slow decay coefficient
    agc_target_level: float = 0.707             # Target normalized peak amplitude

    # Noise Floor Estimation
    noise_window_size: int = 1024               # Samples per sliding noise window
    noise_percentile: float = 50.0              # Median background estimation

    # Pulse Detection & Hysteresis
    snr_margin_db: float = 8.0                  # Threshold margin over noise floor (T_high)
    hysteresis_db: float = 3.0                  # Drop below T_high required to close pulse (T_low)
    min_pulse_samples: int = 10                 # Minimum pulse duration in samples
    min_gap_samples: int = 15                   # Minimum gap to isolate adjacent pulses
    cooldown_samples: int = 5                   # Post-pulse holdoff

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "sample_rate_hz": self.sample_rate_hz,
            "center_frequency_hz": self.center_frequency_hz,
            "bandwidth_hz": self.bandwidth_hz,
            "filter_type": self.filter_type,
            "filter_order": self.filter_order,
            "agc_enabled": self.agc_enabled,
            "agc_attack": self.agc_attack,
            "agc_decay": self.agc_decay,
            "agc_target_level": self.agc_target_level,
            "noise_window_size": self.noise_window_size,
            "noise_percentile": self.noise_percentile,
            "snr_margin_db": self.snr_margin_db,
            "hysteresis_db": self.hysteresis_db,
            "min_pulse_samples": self.min_pulse_samples,
            "min_gap_samples": self.min_gap_samples,
            "cooldown_samples": self.cooldown_samples,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReceiverConfig:
        """Create configuration instance from dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


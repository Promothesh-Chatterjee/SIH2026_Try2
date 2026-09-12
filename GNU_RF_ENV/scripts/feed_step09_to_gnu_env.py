"""
step09_jittered_emitter.grc Ingestion and GNU RF Environment Bridge

Parses step09_jittered_emitter.grc parameters (Multi-emitter dual-band with PRI jitter),
generates the complex IQ stream with multipath and AWGN, processes through
GNU Environment PDW extraction, and registers the scenario for live cognitive scanning.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

# Repository paths
REPO_ROOT = Path(__file__).resolve().parents[2]
GNU_RF_DIR = REPO_ROOT / "GNU_RF_ENV"
COGNITIVE_EW_DIR = REPO_ROOT / "cognitive_ew_smart_scan"

sys.path.insert(0, str(GNU_RF_DIR / "scripts"))
sys.path.insert(0, str(COGNITIVE_EW_DIR / "src"))
sys.path.insert(0, str(REPO_ROOT))

from iq_to_pdw import PDWDetector


def parse_step09_grc(grc_path: Path) -> dict[str, Any]:
    with open(grc_path, "r", encoding="utf-8") as f:
        content = f.read()

    params = {
        "sample_rate": 2e6,
        "emitter1_rf_freq": 3.2e9,
        "emitter1_baseband_freq": 100e3,
        "emitter1_pri": 100e-6,
        "emitter1_pulse_width": 10e-6,
        "emitter1_amplitude": 1.0,
        "emitter2_rf_freq": 8.0e9,
        "emitter2_baseband_freq": -250e3,
        "emitter2_pri": 70e-6,
        "emitter2_pulse_width": 4e-6,
        "emitter2_amplitude": 0.7,
        "jitter_fraction": 0.01,
        "noise_amplitude": 0.05,
    }

    def extract_val(pattern: str, cast_type=float, default=None):
        m = re.search(pattern, content)
        if m:
            try:
                return cast_type(eval(m.group(1), {"__builtins__": None}))
            except Exception:
                try:
                    return cast_type(float(m.group(1)))
                except Exception:
                    pass
        return default

    v = extract_val(r"samp_rate.*?value:\s*'([\d\.eE\+]+)'")
    if v is not None:
        params["sample_rate"] = v

    v = extract_val(r"emitter_rf_freq.*?value:\s*'([\d\.eE\+]+)'")
    if v is not None:
        params["emitter1_rf_freq"] = v

    v = extract_val(r"emitter2_rf_freq.*?value:\s*'([\d\.eE\+]+)'")
    if v is not None:
        params["emitter2_rf_freq"] = v

    v = extract_val(r"pri\b.*?value:\s*'([\d\.eE\-\+]+)'")
    if v is not None:
        params["emitter1_pri"] = v

    v = extract_val(r"emitter2_pri\b.*?value:\s*'([\d\.eE\-\+]+)'")
    if v is not None:
        params["emitter2_pri"] = v

    v = extract_val(r"jitter_fraction.*?value:\s*'([\d\.]+)'")
    if v is not None:
        params["jitter_fraction"] = v

    v = extract_val(r"noise_amplitude.*?value:\s*'([\d\.]+)'")
    if v is not None:
        params["noise_amplitude"] = v

    return params


class Step09MultiEmitterSource:
    """Accurate Python simulation of GNU Radio step09_jittered_emitter flowgraph."""

    def __init__(self, params: dict[str, Any], seed: int = 42):
        self.params = params
        self.sample_rate = float(params["sample_rate"])
        self.rng = np.random.default_rng(seed)

    def generate_stream(self, duration_s: float = 0.5) -> Tuple[np.ndarray, list[dict[str, Any]]]:
        total_samples = int(self.sample_rate * duration_s)
        time_us_horizon = duration_s * 1e6
        dt = 1.0 / self.sample_rate

        # Emitter 1 (Band 6: 3200.1 MHz)
        e1_pri_us = self.params["emitter1_pri"] * 1e6
        e1_pw_us = self.params["emitter1_pulse_width"] * 1e6
        e1_freq_mhz = (self.params["emitter1_rf_freq"] + self.params["emitter1_baseband_freq"]) / 1e6
        e1_bb_hz = self.params["emitter1_baseband_freq"]
        e1_amp = self.params["emitter1_amplitude"]

        # Emitter 2 (Band 16: 7999.75 MHz)
        e2_pri_us = self.params["emitter2_pri"] * 1e6
        e2_pw_us = self.params["emitter2_pulse_width"] * 1e6
        e2_freq_mhz = (self.params["emitter2_rf_freq"] + self.params["emitter2_baseband_freq"]) / 1e6
        e2_bb_hz = self.params["emitter2_baseband_freq"]
        e2_amp = self.params["emitter2_amplitude"]

        jitter_frac = self.params["jitter_fraction"]

        pulse_records = []
        out_iq = np.zeros(total_samples, dtype=np.complex64)
        pulse_id = 0

        # Generate Emitter 1 pulses
        t = float(self.rng.uniform(0.0, e1_pri_us))
        while t < time_us_horizon:
            j = float(self.rng.uniform(-jitter_frac, jitter_frac)) * e1_pri_us
            t_pulse = t + j
            if 0 <= t_pulse < time_us_horizon:
                s_idx = int(round(t_pulse * 1e-6 * self.sample_rate))
                n_samples = int(round(e1_pw_us * 1e-6 * self.sample_rate))
                for s in range(n_samples):
                    if s_idx + s < total_samples:
                        phi = 2.0 * np.pi * e1_bb_hz * (s * dt)
                        out_iq[s_idx + s] += e1_amp * np.exp(1j * phi)
                pulse_records.append({
                    "pulse_id": pulse_id,
                    "toa_us": float(t_pulse),
                    "time_us": float(t_pulse),
                    "frequency_mhz": float(round(e1_freq_mhz, 3)),
                    "pulse_width_us": float(e1_pw_us),
                    "amplitude_db": float(-50.0 + np.random.normal(0, 1.0)),
                    "aoa_deg": float(18.0 + np.random.normal(0, 0.5)),
                    "emitter_id": 1,
                    "source_id": "step09:emitter1",
                })
                pulse_id += 1
            t += e1_pri_us

        # Generate Emitter 2 pulses
        t = float(self.rng.uniform(0.0, e2_pri_us))
        while t < time_us_horizon:
            j = float(self.rng.uniform(-jitter_frac, jitter_frac)) * e2_pri_us
            t_pulse = t + j
            if 0 <= t_pulse < time_us_horizon:
                s_idx = int(round(t_pulse * 1e-6 * self.sample_rate))
                n_samples = int(round(e2_pw_us * 1e-6 * self.sample_rate))
                for s in range(n_samples):
                    if s_idx + s < total_samples:
                        phi = 2.0 * np.pi * e2_bb_hz * (s * dt)
                        out_iq[s_idx + s] += e2_amp * np.exp(1j * phi)
                pulse_records.append({
                    "pulse_id": pulse_id,
                    "toa_us": float(t_pulse),
                    "time_us": float(t_pulse),
                    "frequency_mhz": float(round(e2_freq_mhz, 3)),
                    "pulse_width_us": float(e2_pw_us),
                    "amplitude_db": float(-53.0 + np.random.normal(0, 1.0)),
                    "aoa_deg": float(-35.0 + np.random.normal(0, 0.5)),
                    "emitter_id": 2,
                    "source_id": "step09:emitter2",
                })
                pulse_id += 1
            t += e2_pri_us

        pulse_records.sort(key=lambda p: p["toa_us"])

        # Multipath channel taps (Direct + Delayed)
        # Taps: [1.0 + 0.0j, 0.15 + 0.05j]
        chan_iq = np.convolve(out_iq, [1.0 + 0.0j, 0.15 + 0.05j], mode="same")

        # AWGN channel noise
        noise_amp = self.params["noise_amplitude"]
        noise = (self.rng.normal(0, 1, total_samples) + 1j * self.rng.normal(0, 1, total_samples)) * (noise_amp / np.sqrt(2))
        rx_iq = chan_iq + noise.astype(np.complex64)

        return rx_iq, pulse_records


def feed_step09_to_gnu_env(
    grc_path: Path | None = None,
    duration_s: float = 1.0,
) -> dict[str, Any]:
    if grc_path is None:
        grc_path = GNU_RF_DIR / "flowgraphs" / "step09_jittered_emitter.grc"

    print("=" * 74)
    print(f"[*] Ingesting GNU Radio Flowgraph: {grc_path}")
    params = parse_step09_grc(grc_path)
    e1_f = (params["emitter1_rf_freq"] + params["emitter1_baseband_freq"]) / 1e6
    e2_f = (params["emitter2_rf_freq"] + params["emitter2_baseband_freq"]) / 1e6

    print(f"[*] Extracted step09_jittered_emitter parameters:")
    print(f"    - Sample Rate      : {params['sample_rate'] / 1e6:.1f} MS/s")
    print(f"    - Emitter 1        : {e1_f:.2f} MHz (Band {int(e1_f // 500)}), PRI = {params['emitter1_pri']*1e6:.1f} us, PW = {params['emitter1_pulse_width']*1e6:.1f} us, Jitter = {params['jitter_fraction']*100:.1f}%")
    print(f"    - Emitter 2        : {e2_f:.2f} MHz (Band {int(e2_f // 500)}), PRI = {params['emitter2_pri']*1e6:.1f} us, PW = {params['emitter2_pulse_width']*1e6:.1f} us, Jitter = {params['jitter_fraction']*100:.1f}%")
    print(f"    - Noise Amplitude  : {params['noise_amplitude']}")
    print("=" * 74)

    # 1. Synthesize IQ stream
    source = Step09MultiEmitterSource(params)
    iq_samples, pulse_records = source.generate_stream(duration_s=duration_s)
    print(f"[+] Synthesized {len(iq_samples)} complex64 IQ samples ({duration_s:.2f} s duration).")

    # 2. Feed through GNU Environment: PDW Detection
    detector = PDWDetector(
        sample_rate=params["sample_rate"],
        threshold_db_above_noise=6.0,
        noise_estimation_samples=1000,
        min_pulse_samples=4,
        max_pulse_samples=100,
    )
    detected_pdws = detector.detect_iq(iq_samples)
    print(f"[+] GNU Environment PDWDetector processed stream:")
    print(f"    - Extracted PDWs count   : {len(detected_pdws)}")
    print(f"    - Estimated Noise Floor  : {detector.noise_floor:.4f}")
    print(f"    - Detection Threshold    : {detector.detection_threshold:.4f}")

    # 3. Construct Canonical Scenario GT and save
    episode_dir = GNU_RF_DIR / "p3ac_50k" / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)
    scenario_file = episode_dir / "step09_jittered.gt.json"

    scenario_emitters = [
        {
            "id": "E1_SURVEILLANCE",
            "rf_frequency_mhz": float(round(e1_f, 2)),
            "pulse_width_us": float(params["emitter1_pulse_width"] * 1e6),
            "pri_us": float(params["emitter1_pri"] * 1e6),
            "amplitude": float(params["emitter1_amplitude"]),
            "jitter_fraction": float(params["jitter_fraction"]),
            "band": int(e1_f // 500),
        },
        {
            "id": "E2_TRACKING",
            "rf_frequency_mhz": float(round(e2_f, 2)),
            "pulse_width_us": float(params["emitter2_pulse_width"] * 1e6),
            "pri_us": float(params["emitter2_pri"] * 1e6),
            "amplitude": float(params["emitter2_amplitude"]),
            "jitter_fraction": float(params["jitter_fraction"]),
            "band": int(e2_f // 500),
        },
    ]

    scenario_data = {
        "episode_id": "step09_jittered",
        "generator_version": "step09-GRC-Bridge-v1.0",
        "source_grc": str(grc_path.name),
        "total_pulses": len(pulse_records),
        "emitters": scenario_emitters,
    }

    with open(scenario_file, "w", encoding="utf-8") as f:
        json.dump(scenario_data, f, indent=2)
    print(f"[+] Registered GNU Environment scenario: {scenario_file}")

    raw_pulse_file = GNU_RF_DIR / "data" / "step09_jittered_pulses.json"
    raw_pulse_file.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_pulse_file, "w", encoding="utf-8") as f:
        json.dump(pulse_records, f)
    print(f"[+] Saved raw pulse stream: {raw_pulse_file} ({len(pulse_records)} pulses)")

    return {
        "scenario_name": "step09_jittered",
        "scenario_file": str(scenario_file),
        "pulse_file": str(raw_pulse_file),
        "total_pulses": len(pulse_records),
        "emitters": scenario_emitters,
    }


if __name__ == "__main__":
    feed_step09_to_gnu_env()

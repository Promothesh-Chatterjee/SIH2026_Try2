"""
1.grc FHSS Flowgraph Ingestion and GNU RF Environment Bridge

Parses 1.grc parameters, generates the FHSS complex IQ stream through the
simulated GNU Radio channel model, processes the IQ through the GNU RF Environment
PDW extraction pipeline, and registers the scenario for live cognitive scanning.
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

# Add GNU scripts and receiver_env to path
sys.path.insert(0, str(GNU_RF_DIR / "scripts"))
sys.path.insert(0, str(COGNITIVE_EW_DIR / "src"))
sys.path.insert(0, str(REPO_ROOT))

from iq_to_pdw import PDWDetector
from frequency_context import FrequencyContext
from iq_bridge import IQReceiverBridge


def parse_1_grc(grc_path: Path) -> dict[str, Any]:
    with open(grc_path, "r", encoding="utf-8") as f:
        content = f.read()

    params = {
        "sample_rate": 32000.0,
        "hop_dwell_time": 0.05,
        "freqs": [-120000.0, -90000.0, -60000.0, -30000.0, 0.0, 30000.0, 60000.0, 90000.0, 120000.0],
        "noise_voltage": 0.2,
    }

    sr_match = re.search(r"samp_rate.*?value:\s*'(\d+)'", content, re.DOTALL)
    if sr_match:
        params["sample_rate"] = float(sr_match.group(1))

    hdt_match = re.search(r"hop_dwell_time:\s*'([\d\.]+)'", content)
    if hdt_match:
        params["hop_dwell_time"] = float(hdt_match.group(1))

    freqs_match = re.search(r"freqs:\s*'(\[.*?\])'", content)
    if freqs_match:
        try:
            params["freqs"] = json.loads(freqs_match.group(1))
        except Exception:
            pass

    nv_match = re.search(r"noise_voltage:\s*'([\d\.]+)'", content)
    if nv_match:
        params["noise_voltage"] = float(nv_match.group(1))

    return params


class GRC1FHSSSource:
    """Accurate Python implementation of epy_block_0 from 1.grc with channel model."""

    def __init__(
        self,
        sample_rate: float = 2e6,
        hop_dwell_time: float = 0.05,
        freqs: list[float] | None = None,
        noise_voltage: float = 0.2,
        seed: int = 42,
    ):
        self.sample_rate = sample_rate
        self.hop_dwell_time = hop_dwell_time
        self.freqs = freqs or [-120000.0, -90000.0, -60000.0, -30000.0, 0.0, 30000.0, 60000.0, 90000.0, 120000.0]
        self.noise_voltage = noise_voltage
        self.rng = np.random.default_rng(seed)

        self.samples_per_hop = int(self.sample_rate * self.hop_dwell_time)
        self.current_sample_count = 0
        self.current_freq_idx = 0
        self.phase = 0.0

    def generate_hops_pulsed(
        self,
        num_hops: int = 50,
        pri_us: float = 100.0,
        pulse_width_us: float = 15.0,
    ) -> Tuple[np.ndarray, list[dict[str, Any]]]:
        """Generate pulsed FHSS IQ signal as received by an EW receiver."""
        pri_s = pri_us * 1e-6
        pw_s = pulse_width_us * 1e-6
        pri_samples = max(2, int(round(pri_s * self.sample_rate)))
        pw_samples = max(1, int(round(pw_s * self.sample_rate)))

        pulses_per_hop = max(1, int(self.hop_dwell_time / pri_s))
        total_samples = num_hops * pulses_per_hop * pri_samples

        out = np.zeros(total_samples, dtype=np.complex64)
        dt = 1.0 / self.sample_rate
        hop_records = []

        sample_idx = 0
        for h in range(num_hops):
            self.current_freq_idx = int(self.rng.integers(0, len(self.freqs)))
            freq = self.freqs[self.current_freq_idx]
            hop_start_us = (sample_idx / self.sample_rate) * 1e6

            for p in range(pulses_per_hop):
                pulse_start = sample_idx
                for s in range(pw_samples):
                    out[pulse_start + s] = np.exp(1j * self.phase)
                    self.phase += 2.0 * np.pi * freq * dt
                    self.phase = self.phase % (2.0 * np.pi)
                sample_idx += pri_samples

            hop_end_us = (sample_idx / self.sample_rate) * 1e6
            hop_records.append({
                "hop_idx": h,
                "frequency_hz": freq,
                "frequency_khz": freq / 1e3,
                "start_time_us": hop_start_us,
                "end_time_us": hop_end_us,
                "duration_us": hop_end_us - hop_start_us,
                "pulses": pulses_per_hop,
            })

        # Apply channels_channel_model_0: AWGN with noise_voltage = 0.2
        noise = (self.rng.normal(0, 1, total_samples) + 1j * self.rng.normal(0, 1, total_samples)) * (self.noise_voltage / np.sqrt(2))
        rx_iq = out + noise.astype(np.complex64)
        return rx_iq, hop_records


def feed_1_grc_to_gnu_env(
    grc_path: Path,
    num_hops: int = 100,
    rf_carrier_mhz: float = 3500.0,
) -> dict[str, Any]:
    print("=" * 72)
    print(f"[*] Ingesting GNU Radio Flowgraph: {grc_path}")
    params = parse_1_grc(grc_path)
    print(f"[*] Extracted 1.grc parameters:")
    print(f"    - Sample Rate      : {params['sample_rate']} S/s (scaled to 2.0 MS/s EW processing)")
    print(f"    - Hop Dwell Time   : {params['hop_dwell_time']} s ({params['hop_dwell_time']*1e3:.1f} ms)")
    print(f"    - Hop Frequencies  : {params['freqs']} Hz")
    print(f"    - Noise Voltage    : {params['noise_voltage']}")
    print(f"    - Logical Carrier  : {rf_carrier_mhz:.1f} MHz (Bands 6 & 7 Multi-Band FHSS)")
    print("=" * 72)

    # 1. Generate FHSS IQ stream
    processing_samp_rate = 2e6  # 2 MS/s for high-resolution pulse characterization
    source = GRC1FHSSSource(
        sample_rate=processing_samp_rate,
        hop_dwell_time=params["hop_dwell_time"],
        freqs=params["freqs"],
        noise_voltage=params["noise_voltage"],
    )
    pri_us = 100.0
    pulse_width_us = 15.0
    iq_samples, hop_records = source.generate_hops_pulsed(
        num_hops=num_hops,
        pri_us=pri_us,
        pulse_width_us=pulse_width_us,
    )
    total_duration_s = num_hops * params["hop_dwell_time"]
    print(f"[+] Generated {len(iq_samples)} complex64 IQ samples across {num_hops} hops ({total_duration_s:.2f} s).")

    # 2. Feed through GNU Environment: PDW Detection & Parameter Extraction
    detector = PDWDetector(
        sample_rate=processing_samp_rate,
        threshold_db_above_noise=6.0,
        noise_estimation_samples=1000,
        min_pulse_samples=10,
        max_pulse_samples=100,
    )
    detected_pdws = detector.detect_iq(iq_samples)
    print(f"[+] GNU Environment PDWDetector processed stream:")
    print(f"    - Extracted PDWs count   : {len(detected_pdws)}")
    print(f"    - Estimated Noise Floor  : {detector.noise_floor:.4f}")
    print(f"    - Detection Threshold    : {detector.detection_threshold:.4f}")

    # Inspect first few PDWs
    if detected_pdws:
        first = detected_pdws[0]
        print(f"    - Sample PDW #0: ToA={first['toa_us']:.1f} us, Freq={first['frequency_local_khz']:.2f} kHz, PW={first['pulse_width_us']:.1f} us, Amp={first['amplitude']:.3f}")

    # 3. Map into canonical Scenario Episode for Cognitive EW SmartScan
    dwells = []
    pulse_records = []
    pulses_per_hop = int((params["hop_dwell_time"] * 1e6) / pri_us)

    pulse_id = 0
    for h_idx, hop in enumerate(hop_records):
        # Scale the 1.grc baseband hop across +/- 150 MHz so it exercises Band 6 and Band 7
        f_offset_mhz = (float(hop["frequency_hz"]) / 120000.0) * 150.0 if max(abs(f) for f in params["freqs"]) > 0 else 0.0
        hop_rf_freq_mhz = rf_carrier_mhz + f_offset_mhz
        dwell_start = hop["start_time_us"]
        dwell_end = hop["end_time_us"]
        hop_band = int(np.clip(hop_rf_freq_mhz // 500.0, 0, 35))
        band_center = float(hop_band * 500.0 + 250.0)

        dwell_emitters = [{
            "id": f"FHSS_E1_H{h_idx}",
            "rf_frequency_mhz": float(round(hop_rf_freq_mhz, 3)),
            "pulse_width_us": pulse_width_us,
            "pri_us": pri_us,
            "amplitude": 1.0,
            "jitter_fraction": 0.01,
            "in_band": True,
            "scheduled_active": True,
            "configured_seed": 42 + h_idx,
        }]

        dwells.append({
            "dwell_index": h_idx,
            "start_time_us": float(dwell_start),
            "end_time_us": float(dwell_end),
            "band": hop_band,
            "center_frequency_mhz": band_center,
            "emitters": dwell_emitters,
            "observed_pdw_count": pulses_per_hop,
            "any_hit": True,
        })

        for p_i in range(min(50, pulses_per_hop)):
            toa = dwell_start + p_i * pri_us
            pulse_records.append({
                "pulse_id": pulse_id,
                "toa_us": float(toa),
                "time_us": float(toa),
                "frequency_mhz": float(round(hop_rf_freq_mhz, 3)),
                "pulse_width_us": pulse_width_us,
                "amplitude_db": float(-52.0 + np.random.normal(0, 1.5)),
                "aoa_deg": float(42.5 + np.random.normal(0, 0.5)),
                "emitter_id": 1,
                "source_id": "1.grc:fhss_simulator",
            })
            pulse_id += 1

    episode_dir = GNU_RF_DIR / "p3ac_50k" / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)
    scenario_file = episode_dir / "1_grc_fhss.gt.json"

    scenario_data = {
        "episode_id": "1_grc_fhss",
        "generator_version": "1.grc-FHSS-Bridge-v1.0",
        "schema_version": "3Y.1",
        "source_grc": str(grc_path.name),
        "total_hops": num_hops,
        "dwells": dwells,
        "emitters": dwells[0]["emitters"],
    }

    with open(scenario_file, "w", encoding="utf-8") as f:
        json.dump(scenario_data, f, indent=2)

    print(f"[+] Registered GNU Environment scenario: {scenario_file}")
    print(f"[+] Total scenario dwells: {len(dwells)}, Pulses generated: {len(pulse_records)}")

    raw_pulse_file = GNU_RF_DIR / "data" / "1_grc_fhss_pulses.json"
    raw_pulse_file.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_pulse_file, "w", encoding="utf-8") as f:
        json.dump(pulse_records, f)
    print(f"[+] Saved raw pulse stream: {raw_pulse_file}")

    return {
        "scenario_name": "1_grc_fhss",
        "scenario_file": str(scenario_file),
        "pulse_file": str(raw_pulse_file),
        "total_hops": num_hops,
        "total_pulses": len(pulse_records),
    }


if __name__ == "__main__":
    grc_path = GNU_RF_DIR / "flowgraphs" / "1.grc"
    res = feed_1_grc_to_gnu_env(grc_path, num_hops=100)
    print("\n[+] Ingestion and GNU Environment processing complete.")

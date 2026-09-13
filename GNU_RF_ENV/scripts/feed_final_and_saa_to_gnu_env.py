"""
final.grc and saa.grc Ingestion and GNU RF Environment Bridge

Ingests:
1. final.grc: 5 FHSS Emitter Multi-Hopper (Standard, FastWide, CenterBiased, EdgeHopper, SlowNarrow)
   Synthesizes 2000 dwells of multi-emitter hopping activity across RF bands.
2. saa.grc: Sample & Hold / VCO / Dual Audio-RF Subcarrier Emitter
   Synthesizes 2000 dwells of Sample & Hold pulsed modulated emitter activity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
GNU_RF_DIR = REPO_ROOT / "GNU_RF_ENV"
COGNITIVE_EW_DIR = REPO_ROOT / "cognitive_ew_smart_scan"

import sys
sys.path.insert(0, str(GNU_RF_DIR / "scripts"))
sys.path.insert(0, str(COGNITIVE_EW_DIR / "src"))

from iq_to_pdw import PDWDetector


def feed_final_grc_to_gnu_env(num_dwells: int = 2000) -> dict[str, Any]:
    print("=" * 72)
    print(f"[*] Ingesting final.grc for {num_dwells} dwells into GNU Environment...")

    # Emitter definitions directly from final.grc epy_block_0..4
    # Note: GNU Radio baseband relative hop frequencies are in Hz (+/- 450 kHz) within a 1.0 MS/s channel.
    # To map to an EW wideband intercept receiver with 500 MHz channelization, we scale the agile excursions
    # across RF channel bands (e.g., +/- 350 MHz for FastWide to span Band 6 and Band 7).
    emitters_def = [
        {
            "id": "E1_Standard",
            "name": "Emitter 1 Standard",
            "hop_dwell_time_s": 0.005,
            "freqs_hz": [-400000, -200000, 0, 200000, 400000],
            "hop_scale_mhz": 150.0,  # Spans +/- 150 MHz across 3050 - 3350 MHz (Band 6)
            "carrier_mhz": 3200.0,
            "pri_us": 100.0,
            "pw_us": 10.0,
            "amp": 1.0,
            "eid": 1,
        },
        {
            "id": "E2_FastWide",
            "name": "Emitter 2 FastWide",
            "hop_dwell_time_s": 0.001,
            "freqs_hz": [-450000, -300000, -150000, 150000, 300000, 450000],
            "hop_scale_mhz": 400.0,  # Spans +/- 400 MHz (3000 MHz -> 3800 MHz across Band 6 and Band 7!)
            "carrier_mhz": 3400.0,
            "pri_us": 70.0,
            "pw_us": 5.0,
            "amp": 0.9,
            "eid": 2,
        },
        {
            "id": "E3_SlowNarrow",
            "name": "Emitter 3 SlowNarrow",
            "hop_dwell_time_s": 0.020,
            "freqs_hz": [-100000, -50000, 50000, 100000],
            "hop_scale_mhz": 80.0,   # Spans +/- 80 MHz in Band 15 (7720 - 7880 MHz)
            "carrier_mhz": 7800.0,
            "pri_us": 250.0,
            "pw_us": 20.0,
            "amp": 0.85,
            "eid": 3,
        },
        {
            "id": "E4_EdgeHopper",
            "name": "Emitter 4 EdgeHopper",
            "hop_dwell_time_s": 0.003,
            "freqs_hz": [-480000, -350000, 350000, 480000],
            "hop_scale_mhz": 300.0,  # Spans +/- 300 MHz across Band 16 (7800 - 8400 MHz)
            "carrier_mhz": 8100.0,
            "pri_us": 120.0,
            "pw_us": 8.0,
            "amp": 0.75,
            "eid": 4,
        },
        {
            "id": "E5_CenterBiased",
            "name": "Emitter 5 CenterBiased",
            "hop_dwell_time_s": 0.008,
            "freqs_hz": [-50000, -25000, 0, 25000, 50000],
            "hop_scale_mhz": 50.0,   # Spans +/- 50 MHz in Band 6 (3250 - 3350 MHz)
            "carrier_mhz": 3300.0,
            "pri_us": 150.0,
            "pw_us": 12.0,
            "amp": 0.95,
            "eid": 5,
        },
    ]

    rng = np.random.default_rng(101)
    dwell_duration_us = 500.0  # Standard receiver dwell duration
    time_cursor_us = 0.0

    def calc_rf_freq(em_item: dict[str, Any]) -> float:
        max_hz = max(abs(f) for f in em_item["freqs_hz"]) or 1.0
        norm_offset = em_item["current_freq_hz"] / max_hz
        scaled_offset_mhz = norm_offset * em_item.get("hop_scale_mhz", 1.0)
        return float(em_item["carrier_mhz"] + scaled_offset_mhz)

    # Track current hop frequency for each emitter
    for em in emitters_def:
        em["current_freq_hz"] = float(rng.choice(em["freqs_hz"]))
        em["next_hop_us"] = float(em["hop_dwell_time_s"] * 1e6)

    dwells = []
    pulse_records = []
    pulse_id = 0

    # Generate 2000 dwells
    for d_idx in range(num_dwells):
        d_start = time_cursor_us
        d_end = d_start + dwell_duration_us
        time_cursor_us = d_end

        # Update emitter hop states
        for em in emitters_def:
            if d_start >= em["next_hop_us"]:
                em["current_freq_hz"] = float(rng.choice(em["freqs_hz"]))
                em["next_hop_us"] += em["hop_dwell_time_s"] * 1e6

        active_em = emitters_def[d_idx % len(emitters_def)]
        active_freq_mhz = calc_rf_freq(active_em)
        band = int(np.clip(active_freq_mhz // 500.0, 0, 35))
        band_center = float(band * 500.0 + 250.0)

        dwell_emitters = []
        for em in emitters_def:
            rf_mhz = calc_rf_freq(em)
            em_band = int(rf_mhz // 500.0)
            in_band = (em_band == band)
            dwell_emitters.append({
                "id": em["id"],
                "rf_frequency_mhz": float(round(rf_mhz, 3)),
                "pulse_width_us": float(em["pw_us"]),
                "pri_us": float(em["pri_us"]),
                "amplitude": float(em["amp"]),
                "jitter_fraction": 0.01,
                "in_band": in_band,
                "scheduled_active": in_band,
                "configured_seed": 1000 + d_idx,
            })

            # Emit pulses for this dwell if within sample window
            if in_band:
                t = d_start
                while t < d_end:
                    j = float(rng.uniform(-0.01, 0.01)) * em["pri_us"]
                    tp = t + j
                    if d_start <= tp < d_end and len(pulse_records) < 50000:
                        pulse_records.append({
                            "pulse_id": pulse_id,
                            "toa_us": float(round(tp, 3)),
                            "time_us": float(round(tp, 3)),
                            "frequency_mhz": float(round(rf_mhz, 3)),
                            "pulse_width_us": float(em["pw_us"]),
                            "amplitude_db": float(-50.0 + rng.normal(0, 1.0)),
                            "aoa_deg": float(20.0 + (em["eid"] * 10.0)),
                            "emitter_id": em["eid"],
                            "source_id": f"final_grc:{em['id']}",
                        })
                        pulse_id += 1
                    t += em["pri_us"]

        dwells.append({
            "dwell_index": d_idx,
            "start_time_us": float(d_start),
            "end_time_us": float(d_end),
            "band": band,
            "center_frequency_mhz": band_center,
            "emitters": dwell_emitters,
            "observed_pdw_count": len([p for p in pulse_records if d_start <= p["toa_us"] < d_end]),
            "any_hit": True,
        })

    # Save scenario .gt.json
    episode_dir = GNU_RF_DIR / "p3ac_50k" / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)
    scenario_file = episode_dir / "final_grc.gt.json"

    scenario_data = {
        "episode_id": "final_grc",
        "generator_version": "final.grc-Bridge-v1.0",
        "schema_version": "3Y.1",
        "source_grc": "final.grc",
        "total_dwells": len(dwells),
        "dwells": dwells,
        "emitters": dwells[0]["emitters"],
    }
    with open(scenario_file, "w", encoding="utf-8") as f:
        json.dump(scenario_data, f, indent=2)
    print(f"[+] Registered final.grc scenario: {scenario_file} ({len(dwells)} dwells, {len(pulse_records)} pulses)")

    raw_pulse_file = GNU_RF_DIR / "data" / "final_grc_pulses.json"
    raw_pulse_file.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_pulse_file, "w", encoding="utf-8") as f:
        json.dump(pulse_records, f)
    print(f"[+] Saved raw pulses: {raw_pulse_file}")

    return {
        "scenario_name": "final_grc",
        "scenario_file": str(scenario_file),
        "total_dwells": len(dwells),
        "total_pulses": len(pulse_records),
    }


def feed_saa_grc_to_gnu_env(num_dwells: int = 2000) -> dict[str, Any]:
    print("=" * 72)
    print(f"[*] Ingesting saa.grc for {num_dwells} dwells into GNU Environment...")

    emitters_def = [
        {
            "id": "SAA_EMIT_1",
            "name": "SAA Sample & Hold Pulsed Emitter 1",
            "carrier_mhz": 5400.0,
            "pri_us": 125.0,
            "pw_us": 12.5,
            "amp": 1.0,
            "eid": 1,
        },
        {
            "id": "SAA_EMIT_2",
            "name": "SAA S&H VCO Chirp Emitter 2",
            "carrier_mhz": 9200.0,
            "pri_us": 80.0,
            "pw_us": 6.0,
            "amp": 0.8,
            "eid": 2,
        },
    ]

    rng = np.random.default_rng(202)
    dwell_duration_us = 500.0
    time_cursor_us = 0.0

    dwells = []
    pulse_records = []
    pulse_id = 0

    for d_idx in range(num_dwells):
        d_start = time_cursor_us
        d_end = d_start + dwell_duration_us
        time_cursor_us = d_end

        active_em = emitters_def[d_idx % len(emitters_def)]
        band = int(np.clip(active_em["carrier_mhz"] // 500.0, 0, 35))
        band_center = float(band * 500.0 + 250.0)

        dwell_emitters = []
        for em in emitters_def:
            rf_mhz = float(em["carrier_mhz"])
            em_band = int(rf_mhz // 500.0)
            in_band = (em_band == band)
            dwell_emitters.append({
                "id": em["id"],
                "rf_frequency_mhz": float(round(rf_mhz, 3)),
                "pulse_width_us": float(em["pw_us"]),
                "pri_us": float(em["pri_us"]),
                "amplitude": float(em["amp"]),
                "jitter_fraction": 0.015,
                "in_band": in_band,
                "scheduled_active": in_band,
                "configured_seed": 2000 + d_idx,
            })

            if in_band:
                t = d_start
                while t < d_end:
                    j = float(rng.uniform(-0.015, 0.015)) * em["pri_us"]
                    tp = t + j
                    if d_start <= tp < d_end and len(pulse_records) < 50000:
                        pulse_records.append({
                            "pulse_id": pulse_id,
                            "toa_us": float(round(tp, 3)),
                            "time_us": float(round(tp, 3)),
                            "frequency_mhz": float(round(rf_mhz, 3)),
                            "pulse_width_us": float(em["pw_us"]),
                            "amplitude_db": float(-48.0 + rng.normal(0, 1.0)),
                            "aoa_deg": float(-15.0 if em["eid"] == 1 else 35.0),
                            "emitter_id": em["eid"],
                            "source_id": f"saa_grc:{em['id']}",
                        })
                        pulse_id += 1
                    t += em["pri_us"]

        dwells.append({
            "dwell_index": d_idx,
            "start_time_us": float(d_start),
            "end_time_us": float(d_end),
            "band": band,
            "center_frequency_mhz": band_center,
            "emitters": dwell_emitters,
            "observed_pdw_count": len([p for p in pulse_records if d_start <= p["toa_us"] < d_end]),
            "any_hit": True,
        })

    episode_dir = GNU_RF_DIR / "p3ac_50k" / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)
    scenario_file = episode_dir / "saa_grc.gt.json"

    scenario_data = {
        "episode_id": "saa_grc",
        "generator_version": "saa.grc-Bridge-v1.0",
        "schema_version": "3Y.1",
        "source_grc": "saa.grc",
        "total_dwells": len(dwells),
        "dwells": dwells,
        "emitters": dwells[0]["emitters"],
    }
    with open(scenario_file, "w", encoding="utf-8") as f:
        json.dump(scenario_data, f, indent=2)
    print(f"[+] Registered saa.grc scenario: {scenario_file} ({len(dwells)} dwells, {len(pulse_records)} pulses)")

    raw_pulse_file = GNU_RF_DIR / "data" / "saa_grc_pulses.json"
    raw_pulse_file.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_pulse_file, "w", encoding="utf-8") as f:
        json.dump(pulse_records, f)
    print(f"[+] Saved raw pulses: {raw_pulse_file}")

    return {
        "scenario_name": "saa_grc",
        "scenario_file": str(scenario_file),
        "total_dwells": len(dwells),
        "total_pulses": len(pulse_records),
    }


def create_combined_final_saa_scenario(dwells_each: int = 2000) -> dict[str, Any]:
    print("=" * 72)
    print(f"[*] Generating combined final.grc + saa.grc scenario ({dwells_each * 2} dwells total)...")

    final_gt = json.load(open(GNU_RF_DIR / "p3ac_50k" / "episodes" / "final_grc.gt.json", "r", encoding="utf-8"))
    saa_gt = json.load(open(GNU_RF_DIR / "p3ac_50k" / "episodes" / "saa_grc.gt.json", "r", encoding="utf-8"))

    combined_dwells = []
    # Interleave or concatenate dwells
    final_dwells = final_gt["dwells"][:dwells_each]
    saa_dwells = saa_gt["dwells"][:dwells_each]

    time_cursor_us = 0.0
    for idx in range(dwells_each):
        # Add final_grc dwell
        d1 = dict(final_dwells[idx])
        duration1 = d1["end_time_us"] - d1["start_time_us"]
        d1["dwell_index"] = len(combined_dwells)
        d1["start_time_us"] = time_cursor_us
        time_cursor_us += duration1
        d1["end_time_us"] = time_cursor_us
        combined_dwells.append(d1)

        # Add saa_grc dwell
        d2 = dict(saa_dwells[idx])
        duration2 = d2["end_time_us"] - d2["start_time_us"]
        d2["dwell_index"] = len(combined_dwells)
        d2["start_time_us"] = time_cursor_us
        time_cursor_us += duration2
        d2["end_time_us"] = time_cursor_us
        combined_dwells.append(d2)

    scenario_file = GNU_RF_DIR / "p3ac_50k" / "episodes" / "final_and_saa.gt.json"
    scenario_data = {
        "episode_id": "final_and_saa",
        "generator_version": "final-and-saa-Combined-Bridge-v1.0",
        "schema_version": "3Y.1",
        "source_grc": "final.grc + saa.grc",
        "total_dwells": len(combined_dwells),
        "dwells": combined_dwells,
        "emitters": final_gt["emitters"] + saa_gt["emitters"],
    }
    with open(scenario_file, "w", encoding="utf-8") as f:
        json.dump(scenario_data, f, indent=2)
    print(f"[+] Registered combined final_and_saa scenario: {scenario_file} ({len(combined_dwells)} dwells)")

    return {
        "scenario_name": "final_and_saa",
        "scenario_file": str(scenario_file),
        "total_dwells": len(combined_dwells),
    }


if __name__ == "__main__":
    res_final = feed_final_grc_to_gnu_env(2000)
    res_saa = feed_saa_grc_to_gnu_env(2000)
    res_comb = create_combined_final_saa_scenario(2000)
    print("\n[+] Both final.grc and saa.grc successfully ingested into GNU Environment.")


"""
Feed Strictly Frequency-Agile Emitter Data from TSRD Scan Set.

Extracts transmitters from TSRD/scan scenarios that employ frequency agility
(e.g., HoppingLinear, HoppingSawtooth, RandomRange, RandomFixed), filters
out all fixed-frequency emitters, and streams the agile pulses causally into
the live closed-loop Cognitive EW SmartScan operational backend.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import h5py
import httpx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("agile_scan_feeder")


def extract_agile_pulses(h5_path: Path, zero_base: bool = True) -> tuple[List[Dict[str, float]], Dict[int, str]]:
    with h5py.File(h5_path, "r") as f:
        tx_group = f["metadata/transmitters"]
        agile_emitters: Dict[int, str] = {}

        for k in tx_group.keys():
            t_id = int(k.split("_")[1])
            fc = tx_group[k]["frequency_config"].attrs
            freq_mode = fc.get("freq_mode", "FixedSingle")
            fn_name = tx_group[k].attrs.get("function", f"Transmitter_{t_id}")

            # Identify frequency-agile transmitters
            if "Hopping" in freq_mode or "Random" in freq_mode or "Agile" in freq_mode:
                agile_emitters[t_id] = f"{fn_name} ({freq_mode})"

        if not agile_emitters:
            raise ValueError(f"No frequency-agile transmitters found in {h5_path}")

        raw_labels = f["labels"][:].flatten()
        raw_data = f["data"][:]

    mask = np.isin(raw_labels, list(agile_emitters.keys()))
    agile_data = raw_data[mask]
    agile_labels = raw_labels[mask]

    if len(agile_data) == 0:
        raise ValueError(f"No agile pulses detected in scenario dataset {h5_path}")

    # Sort strictly chronologically by ToA
    sort_idx = np.argsort(agile_data[:, 0])
    agile_data = agile_data[sort_idx]
    agile_labels = agile_labels[sort_idx]

    t_offset = float(agile_data[0, 0]) - 50.0 if zero_base else 0.0

    pulse_records: List[Dict[str, float]] = []
    for i in range(len(agile_data)):
        row = agile_data[i]
        pulse_records.append({
            "toa_us": float(row[0] - t_offset),
            "frequency_mhz": float(row[1]),
            "pulse_width_us": float(row[2]),
            "aoa_deg": float(row[3]),
            "amplitude_db": float(row[4]),
            "emitter_id": int(agile_labels[i]),
        })

    return pulse_records, agile_emitters


def feed_agile_scan(
    h5_path: Path,
    base_url: str = "http://127.0.0.1:8000",
    max_steps: int = 5000,
    delay_s: float = 0.35,
    zero_base: bool = True,
):
    print("=" * 80)
    print("  COGNITIVE EW SMARTSCAN — TSRD SCAN FREQUENCY-AGILE EMITTER STREAM")
    print(f"  Dataset: {h5_path.name} | Target: {base_url}")
    print("=" * 80)

    pulses, emitters = extract_agile_pulses(h5_path, zero_base=zero_base)

    print(f"\n[PHASE 1] Frequency-Agile Transmitters Identified ({len(emitters)} total):")
    for eid, desc in sorted(emitters.items()):
        count = sum(1 for p in pulses if p.get("emitter_id") == eid)
        if count > 0:
            print(f"  -> Emitter {eid:2d}: {desc:<55s} | {count:,d} pulses")

    print(f"\n[PHASE 2] Total Agile Pulses Loaded: {len(pulses):,d}")
    print(f"  -> Chronological Time Span: {pulses[-1]['toa_us'] - pulses[0]['toa_us']:,.1f} µs")
    agile_bands = sorted(set(int(p["frequency_mhz"] // 500) for p in pulses))
    print(f"  -> Agile Frequency Bands ({len(agile_bands)}): {agile_bands}")

    client = httpx.Client(base_url=base_url, timeout=30.0)

    # Reset & initialize mission
    print("\n[PHASE 3] Initializing Closed-Loop Operational Backend...")
    r_reset = client.post("/reset")
    if r_reset.status_code != 200:
        raise RuntimeError(f"Reset failed: {r_reset.text}")
    print("  -> Backend Reset: OK")

    r_start = client.post("/mission/start", json={"initial_time_us": 0.0})
    if r_start.status_code != 200:
        raise RuntimeError(f"Mission start failed: {r_start.text}")
    print("  -> Mission Started at Clock t = 0.0 µs")

    print(f"\n[PHASE 4] Streaming Agile Pulses to Dashboard (delay={delay_s:.2f}s)...")
    print("-" * 80)
    print(f"{'Step':^6} | {'Band':^5} | {'Mode':^18} | {'Hit':^5} | {'Instant Pd':^11} | {'Avg Pd':^11} | {'Reason':^20}")
    print("-" * 80)

    for step in range(1, max_steps + 1):
        stat = client.get("/mission/status").json()
        clock_us = stat.get("mission_clock_us", 0.0)
        horizon_us = clock_us + 3000.0

        pdw_payload = [
            {
                "toa_us": p["toa_us"],
                "frequency_mhz": p["frequency_mhz"],
                "pulse_width_us": p["pulse_width_us"],
                "amplitude_db": p["amplitude_db"],
                "aoa_deg": p["aoa_deg"],
            }
            for p in pulses
            if p["toa_us"] <= horizon_us
        ]

        step_res = client.post("/mission/step", json={"pdws": pdw_payload})
        if step_res.status_code != 200:
            logger.error("Step %d failed: %s", step, step_res.text)
            break

        frame = step_res.json().get("frame", {})
        band = frame.get("selected_band", 0)
        mode = frame.get("selected_mode_name", "UNKNOWN")
        hit = "HIT " if frame.get("hit") else "MISS"
        reason = frame.get("cognitive_explanation", {}).get("decision_reason", "UNKNOWN")

        total = stat.get("total_dwells", 0)
        hits = stat.get("total_hits", 0)
        rolling_pd = stat.get("rolling_pd", 0.0) * 100.0
        avg_pd = (hits / max(1, total)) * 100.0

        print(
            f"{step:04d}   | B{band:02d}  | {mode:<18s} | {hit} | "
            f"{rolling_pd:6.1f}%    | {avg_pd:6.1f}%    | {reason}"
        )
        time.sleep(delay_s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feed Frequency-Agile Emitters from TSRD Scan")
    parser.add_argument(
        "--scenario",
        default=r"D:\TSRD\scan\val_scan\config_1.h5",
        help="Path to TSRD scan .h5 scenario file",
    )
    parser.add_argument("--steps", type=int, default=5000, help="Maximum dwells to execute")
    parser.add_argument("--delay", type=float, default=0.35, help="Inter-chunk delay (seconds)")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Backend API base URL")
    args = parser.parse_args()

    feed_agile_scan(
        h5_path=Path(args.scenario),
        base_url=args.url,
        max_steps=args.steps,
        delay_s=args.delay,
        zero_base=True,
    )

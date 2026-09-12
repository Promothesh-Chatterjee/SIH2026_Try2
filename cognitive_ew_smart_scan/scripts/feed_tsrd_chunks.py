"""
Feed Real TSRD Chunks to Live Closed-Loop ML Scheduler.

Ingests authentic radar pulse data from official Turing Synthetic Radar Dataset (TSRD)
HDF5 files, divides the pulse stream into sequential real-time chunks, and feeds
them step-by-step into the live Cognitive EW SmartScan operational backend.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import httpx
import numpy as np

# Ensure cognitive_ew_smart_scan is in path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment.scenario_generator import load_h5_records

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tsrd_chunk_feeder")


def feed_tsrd_chunks(
    h5_path: Path,
    base_url: str = "http://127.0.0.1:8000",
    max_steps: int = 50,
    chunk_pulse_size: int = 50,
    step_delay_s: float = 0.05,
    output_report: Path | None = None,
) -> Dict[str, Any]:
    print("=" * 80)
    print("  COGNITIVE EW SMARTSCAN — REAL TSRD CHUNK EVALUATION")
    print(f"  Dataset: {h5_path.name} | Target Server: {base_url}")
    print("=" * 80)

    if not h5_path.exists():
        raise FileNotFoundError(f"TSRD file not found: {h5_path}")

    # 1. Load authentic pulse stream (GNU parsed data or TSRD HDF5)
    print(f"\n[PHASE 1] Ingesting Authentic Scenario Dataset: '{h5_path.name}'...")
    from src.environment.scenario_generator import DEFAULT_GNU_DATA_PATH, load_gnu_records

    if h5_path.suffix in [".json", ".npz"] or "GNU_RF_ENV" in str(h5_path):
        records = load_gnu_records(h5_path, max_pulses=max_steps * chunk_pulse_size * 2)
    else:
        records = load_h5_records(h5_path, max_pulses=max_steps * chunk_pulse_size * 2)
    if not records:
        raise ValueError(f"No valid pulses extracted from {h5_path}")

    total_duration_us = records[-1].toa_us - records[0].toa_us
    print(f"  -> Total Extracted Pulses : {len(records):,}")
    print(f"  -> Time Horizon Span      : {total_duration_us:,.1f} µs ({total_duration_us / 1000.0:.2f} ms)")
    print(f"  -> Emitter Count in File  : {len(set(r.emitter_id for r in records))}")

    client = httpx.Client(base_url=base_url, timeout=30.0)

    # 2. Reset and start mission
    print("\n[PHASE 2] Initializing & Synchronizing Closed-Loop Mission...")
    res = client.post("/reset")
    if res.status_code != 200:
        raise RuntimeError(f"Failed to reset backend: {res.text}")
    print("  -> Backend Reset: OK")

    res = client.post("/mission/start", json={"initial_time_us": 0.0})
    if res.status_code != 200:
        raise RuntimeError(f"Failed to start mission: {res.text}")
    print("  -> Mission Started at Clock t = 0.0 µs")

    # 3. Stream real chunks through the ML Scheduler
    print(f"\n[PHASE 3] Streaming {max_steps} Consecutive Chunks ({chunk_pulse_size} pulses/chunk)...")
    print("-" * 80)
    print(f"{'Chunk':^6} | {'Clock (µs)':^11} | {'Band':^5} | {'Mode':^16} | {'Hit':^5} | {'Hits/Tot':^9} | {'Pd (%)':^7} | {'Reason':^18}")
    print("-" * 80)

    pulse_idx = 0
    chunk_metrics: List[Dict[str, Any]] = []
    hits_so_far = 0
    latencies: List[float] = []
    cycle_latencies_ms: List[float] = []
    reasons_count: Dict[str, int] = {}
    modes_count: Dict[str, int] = {}
    bands_count: Dict[int, int] = {}

    for step in range(1, max_steps + 1):
        t_start = time.perf_counter()

        # Ingest pulses causally up to current mission clock + dwell horizon (e.g. 2500 us)
        # We also keep previously ingested pulses in buffer (the receiver adapter discards past pulses)
        status_res = client.get("/mission/status")
        current_clock_us = status_res.json().get("mission_clock_us", 0.0)

        # Buffer pulses from beginning up to current_clock + 3000 us
        horizon_us = current_clock_us + 3000.0
        relevant_pulses = [r for r in records if r.toa_us <= horizon_us]

        pdw_payload = [
            {
                "toa_us": float(r.toa_us),
                "frequency_mhz": float(r.frequency_mhz),
                "pulse_width_us": float(r.pulse_width_us),
                "amplitude_db": float(r.amplitude_db),
                "aoa_deg": float(r.aoa_deg),
            }
            for r in relevant_pulses
        ]

        # Post chunk to live backend
        step_res = client.post("/mission/step", json={"pdws": pdw_payload})
        if step_res.status_code != 200:
            logger.error("Error at step %d: %s", step, step_res.text)
            break

        step_duration_ms = (time.perf_counter() - t_start) * 1000.0
        cycle_latencies_ms.append(step_duration_ms)

        frame = step_res.json().get("frame", {})
        band = frame.get("selected_band", 0)
        mode = frame.get("mode_name", "UNKNOWN")
        hit = frame.get("hit", False)
        reason = frame.get("cognitive_explanation", {}).get("decision_reason", "UNKNOWN")
        clock_us = frame.get("dwell_end_us", 0.0)

        if hit:
            hits_so_far += 1
            lat = frame.get("intercept_time_us", 0.0) - frame.get("dwell_start_us", 0.0)
            if lat > 0:
                latencies.append(lat)

        current_pd = (hits_so_far / step) * 100.0
        reasons_count[reason] = reasons_count.get(reason, 0) + 1
        modes_count[mode] = modes_count.get(mode, 0) + 1
        bands_count[band] = bands_count.get(band, 0) + 1

        chunk_metrics.append({
            "chunk_index": step,
            "clock_us": clock_us,
            "selected_band": band,
            "mode_name": mode,
            "hit": hit,
            "num_detections": frame.get("num_detections", 0),
            "reason": reason,
            "cycle_ms": step_duration_ms,
        })

        hit_str = " HIT " if hit else "MISS "
        print(f"{step:^6d} | {clock_us:^11.1f} | {band:^5d} | {mode:^16} | {hit_str:^5} | {hits_so_far:^4d}/{step:^4d} | {current_pd:^7.1f} | {reason:^18}")

        if step_delay_s > 0:
            time.sleep(step_delay_s)

    print("-" * 80)

    # 4. Finalize mission
    client.post("/mission/stop")
    print(f"\n[PHASE 4] Mission Halted. Evaluation Summary on Real TSRD Pulses:")
    final_pd = (hits_so_far / max_steps) * 100.0
    med_lat = float(np.median(latencies)) if latencies else 0.0
    mean_cycle_ms = float(np.mean(cycle_latencies_ms))
    p95_cycle_ms = float(np.percentile(cycle_latencies_ms, 95))

    print(f"  -> Total Real Dwells Executed    : {max_steps}")
    print(f"  -> Intercepted Pulses / Events  : {hits_so_far}")
    print(f"  -> Overall Detection Rate (Pd)  : {final_pd:.2f}%")
    print(f"  -> Median Intercept Latency     : {med_lat:.1f} µs")
    print(f"  -> Mean Decision Cycle Latency  : {mean_cycle_ms:.2f} ms (P95: {p95_cycle_ms:.2f} ms)")
    print(f"  -> Distinct Bands Monitored     : {len(bands_count)} / 36")
    print(f"  -> Cognitive Reason Breakdown   : {dict(reasons_count)}")
    print(f"  -> Dwell Mode Allocation        : {dict(modes_count)}")

    report = {
        "scenario": h5_path.name,
        "max_steps": max_steps,
        "chunk_pulse_size": chunk_pulse_size,
        "total_hits": hits_so_far,
        "pd_pct": final_pd,
        "median_latency_us": med_lat,
        "mean_cycle_latency_ms": mean_cycle_ms,
        "p95_cycle_latency_ms": p95_cycle_ms,
        "distinct_bands": len(bands_count),
        "cognitive_reasons": reasons_count,
        "dwell_modes": modes_count,
        "chunk_metrics": chunk_metrics,
    }

    if output_report:
        output_report.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  -> Report Saved to: {output_report}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feed Real TSRD Chunks to Live Cognitive EW Backend")
    from src.environment.scenario_generator import DEFAULT_GNU_DATA_PATH, DEFAULT_GNU_DIR

    parser.add_argument(
        "--scenario",
        default=str(DEFAULT_GNU_DATA_PATH),
        help=f"Path to GNU parsed scenario (.gt.json, default: {DEFAULT_GNU_DATA_PATH.name}) or TSRD .h5",
    )
    parser.add_argument("--steps", type=int, default=50, help="Number of chunks/dwells to execute")
    parser.add_argument("--chunk-size", type=int, default=50, help="Pulses fed per chunk")
    parser.add_argument("--delay", type=float, default=0.03, help="Inter-chunk delay (seconds) for UI viewing")
    parser.add_argument("--output", default="results/real_tsrd_chunk_evaluation.json", help="Path to output report")
    args = parser.parse_args()

    scen_path = Path(args.scenario)
    if not scen_path.exists():
        # Try finding in GNU episodes directory
        alt_gnu = DEFAULT_GNU_DIR / f"{scen_path.name}"
        if not alt_gnu.exists() and not str(scen_path).endswith(".gt.json"):
            alt_gnu = DEFAULT_GNU_DIR / f"{scen_path.stem}.gt.json"
        if alt_gnu.exists():
            scen_path = alt_gnu
        else:
            alt_tsrd = Path(r"D:\TSRD\stare\val_stare") / f"{args.scenario}.h5"
            if alt_tsrd.exists():
                scen_path = alt_tsrd

    feed_tsrd_chunks(
        h5_path=scen_path,
        max_steps=args.steps,
        chunk_pulse_size=args.chunk_size,
        step_delay_s=args.delay,
        output_report=Path(args.output),
    )

"""
Continuous Live Data Feeder for Cognitive EW SmartScan.

Streams authentic Agile radar scenario pulses (AG-01 to AG-10) or TSRD HDF5 files
into the running FastAPI backend server at http://127.0.0.1:8000.
Feeds dwell cycles causally, driving live WebSocket and REST telemetry updates
to the frontend dashboard.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_agile_benchmark import generate_agile_scenario

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stream_live_feed")


def build_scenario_records(scenario_id: str, time_horizon_us: float = 600_000.0) -> List[Dict[str, Any]]:
    """Generate or load incident RF pulse train."""
    h5_candidate = Path(scenario_id)
    if h5_candidate.exists() and h5_candidate.suffix == ".h5":
        from src.environment.scenario_generator import load_h5_records
        records, _ = load_h5_records(h5_candidate, max_pulses=50000)
        return [
            {
                "toa_us": float(r.toa_us),
                "frequency_mhz": float(r.frequency_mhz),
                "pulse_width_us": float(r.pulse_width_us),
                "amplitude_db": float(r.amplitude_db),
                "aoa_deg": float(r.aoa_deg),
            }
            for r in records
        ]

    # Generate Agile Scenario (AG-01 through AG-10)
    scen = scenario_id if scenario_id.startswith("AG-") else "AG-04"
    pulse_records = generate_agile_scenario(scen, time_horizon_us=time_horizon_us, seed=42)
    # Sort pulses strictly by TOA
    sorted_recs = sorted(pulse_records, key=lambda r: r.toa_us)
    return [
        {
            "toa_us": float(r.toa_us),
            "frequency_mhz": float(r.frequency_mhz),
            "pulse_width_us": float(r.pulse_width_us),
            "amplitude_db": float(r.amplitude_db),
            "aoa_deg": float(r.aoa_deg),
        }
        for r in sorted_recs
    ]


def stream_live_feed(
    scenario_id: str = "AG-04",
    base_url: str = "http://127.0.0.1:8000",
    delay_s: float = 0.20,
    max_steps: int = 0,
    infinite: bool = True,
):
    print("=" * 82)
    print("  COGNITIVE EW SMARTSCAN — REAL-TIME OPERATIONAL RF STREAM FEEDER")
    print(f"  Target Server: {base_url} | Scenario: {scenario_id} | Cadence: {1.0 / delay_s:.1f} Hz")
    print("=" * 82)

    client = httpx.Client(base_url=base_url, timeout=10.0)

    # Check backend health
    try:
        health_res = client.get("/health")
        if health_res.status_code != 200:
            print(f"[ERROR] Backend not healthy: {health_res.text}")
            return
        health = health_res.json()
        print(f"  -> Backend Status: {health.get('status')} (Controller Ready: {health.get('mission_controller_ready')})")
    except Exception as exc:
        print(f"[ERROR] Could not connect to {base_url}: {exc}")
        return

    # Start or initialize mission
    try:
        client.post("/mission/start", json={"initial_time_us": 0.0})
        print("  -> Closed-Loop Mission Started at Clock t = 0.0 us")
    except Exception as exc:
        print(f"[WARNING] Could not start mission: {exc}")

    # Generate pulse train
    horizon_us = 1_000_000.0  # 1 second of RF mission time per block
    print(f"  -> Generating Incident RF Pulse Train for '{scenario_id}'...")
    pulses = build_scenario_records(scenario_id, time_horizon_us=horizon_us)
    print(f"  -> Loaded {len(pulses):,d} Incident Radar Pulses across bands.")

    print("\n" + "-" * 82)
    print(f"{'Step':^6} | {'Clock (us)':^11} | {'Band':^5} | {'Mode':^16} | {'Status':^6} | {'Pd (%)':^7} | {'Decision Reason':^22}")
    print("-" * 82)

    step_count = 0
    hits = 0
    loop_count = 0

    try:
        while True:
            # Query current mission clock
            try:
                status_res = client.get("/mission/status")
                clock_us = status_res.json().get("mission_clock_us", 0.0)
            except Exception:
                clock_us = float(step_count * 1000.0)

            # Extract pulses in current causal lookahead window (current clock + 3500 us)
            dwell_lookahead_us = clock_us + 3500.0
            window_pulses = [p for p in pulses if p["toa_us"] <= dwell_lookahead_us]

            # If horizon reached in pulse buffer, advance or regenerate
            if not window_pulses or dwell_lookahead_us > horizon_us:
                if not infinite and max_steps > 0 and step_count >= max_steps:
                    break
                # Loop scenario by resetting mission or advancing horizon
                loop_count += 1
                client.post("/mission/start", json={"initial_time_us": 0.0})
                clock_us = 0.0
                dwell_lookahead_us = 3500.0
                window_pulses = [p for p in pulses if p["toa_us"] <= dwell_lookahead_us]
                print(f"  --- Scenario Loop #{loop_count} Re-synchronized ---")

            # Post operational dwell step
            step_count += 1
            try:
                res = client.post("/mission/step", json={"pdws": window_pulses})
                if res.status_code == 200:
                    data = res.json().get("frame", {})
                    band = data.get("selected_band", 0)
                    mode = data.get("mode_name", "NORMAL_DWELL")
                    hit = data.get("hit", False)
                    end_us = data.get("dwell_end_us", 0.0)
                    reason = data.get("cognitive_explanation", {}).get("decision_reason", "DRQN Policy")
                    rolling_pd = (data.get("system_metrics", {}).get("rolling_pd", 0.0)) * 100.0

                    if hit:
                        hits += 1

                    hit_label = "HIT" if hit else "SEARCH"
                    reason_short = (reason[:20] + "..") if len(reason) > 22 else reason
                    print(f"{step_count:^6d} | {end_us:^11.1f} | B{band:02d}  | {mode:^16} | {hit_label:^6} | {rolling_pd:^7.1f} | {reason_short:22s}")
                else:
                    print(f"[WARN] Step returned {res.status_code}: {res.text}")
            except Exception as exc:
                print(f"[WARN] Step error: {exc}")

            if max_steps > 0 and step_count >= max_steps:
                break

            time.sleep(delay_s)

    except KeyboardInterrupt:
        print("\n[INFO] Streaming stopped by user.")
    finally:
        try:
            client.post("/mission/stop")
        except Exception:
            pass
        print(f"\n[DONE] Finished {step_count} dwell cycles. Final Intercept Rate: {(hits / max(1, step_count))*100:.1f}%.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live RF Data Feeder for Cognitive EW SmartScan")
    parser.add_argument("--scenario", default="AG-04", help="Agile threat scenario (AG-01 to AG-10) or path to .h5")
    parser.add_argument("--delay", type=float, default=0.20, help="Cadence delay between dwell cycles (seconds)")
    parser.add_argument("--steps", type=int, default=0, help="Max steps (0 for continuous stream)")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="FastAPI backend URL")
    args = parser.parse_args()

    stream_live_feed(
        scenario_id=args.scenario,
        base_url=args.base_url,
        delay_s=args.delay,
        max_steps=args.steps,
        infinite=(args.steps == 0),
    )

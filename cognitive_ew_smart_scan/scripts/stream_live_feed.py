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
from src.environment.scenario_generator import DEFAULT_GNU_DATA_PATH, DEFAULT_GNU_DIR, load_gnu_records

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stream_live_feed")


def build_scenario_records(scenario_id: str | None = None, time_horizon_us: float = 600_000.0) -> List[Dict[str, Any]]:
    """Generate or load incident RF pulse train. Defaults to GNU parsed data."""
    scen_str = str(scenario_id) if scenario_id else str(DEFAULT_GNU_DATA_PATH)

    # 1. Check if scenario refers to GNU parsed data (.gt.json, .npz, episode ID, or default)
    gnu_path = Path(scen_str)
    if not gnu_path.exists() and (scen_str.startswith("EP") or scen_str.startswith("ep")):
        cand = DEFAULT_GNU_DIR / f"{scen_str}.gt.json"
        if cand.exists():
            gnu_path = cand

    if gnu_path.exists() and (gnu_path.suffix in [".json", ".npz"] or "GNU_RF_ENV" in str(gnu_path)):
        logger.info("Loading incident pulses from GNU parsed dataset: %s", gnu_path)
        records = load_gnu_records(gnu_path, time_horizon_us=time_horizon_us)
        return [
            {
                "toa_us": float(r.toa_us),
                "frequency_mhz": float(r.frequency_mhz),
                "pulse_width_us": float(r.pulse_width_us),
                "amplitude_db": float(r.amplitude_db),
                "aoa_deg": float(r.aoa_deg),
                "emitter_id": r.emitter_id,
            }
            for r in records
        ]

    # 2. Check for TSRD HDF5 file
    h5_candidate = Path(scen_str)
    if h5_candidate.exists() and h5_candidate.suffix == ".h5":
        from src.environment.scenario_generator import load_h5_records
        records = load_h5_records(h5_candidate, max_pulses=50000)
        return [
            {
                "toa_us": float(r.toa_us),
                "frequency_mhz": float(r.frequency_mhz),
                "pulse_width_us": float(r.pulse_width_us),
                "amplitude_db": float(r.amplitude_db),
                "aoa_deg": float(r.aoa_deg),
                "emitter_id": r.emitter_id,
            }
            for r in records
        ]

    # 3. Fallback: Generate Agile Scenario (AG-01 through AG-10)
    scen = scen_str if scen_str.startswith("AG-") else "AG-04"
    logger.info("Generating agile scenario pulses: %s", scen)
    pulse_records = generate_agile_scenario(scen, time_horizon_us=time_horizon_us, seed=42)
    sorted_recs = sorted(pulse_records, key=lambda r: r.toa_us)
    return [
        {
            "toa_us": float(r.toa_us),
            "frequency_mhz": float(r.frequency_mhz),
            "pulse_width_us": float(r.pulse_width_us),
            "amplitude_db": float(r.amplitude_db),
            "aoa_deg": float(r.aoa_deg),
            "emitter_id": getattr(r, "emitter_id", 0),
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

            # Lightweight causal lookahead window: modulo scenario duration
            t_mod = clock_us % horizon_us
            offset = (clock_us // horizon_us) * horizon_us
            window_pulses = [
                {
                    **p,
                    "toa_us": float(p["toa_us"] + offset),
                    "time_us": float(p["toa_us"] + offset),
                }
                for p in pulses
                if t_mod - 500.0 <= p["toa_us"] <= t_mod + 3500.0
            ]

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
    parser.add_argument(
        "--scenario",
        default=str(DEFAULT_GNU_DATA_PATH),
        help=f"Path to GNU parsed scenario (.gt.json, default: {DEFAULT_GNU_DATA_PATH.name}), TSRD .h5, or Agile scenario ID",
    )
    parser.add_argument("--delay", type=float, default=0.15, help="Cadence delay between dwell cycles (seconds)")
    parser.add_argument("--steps", type=int, default=0, help="Max steps (0 for continuous stream)")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="FastAPI backend URL")
    parser.add_argument("--backend-worker", action="store_true", help="Start continuous live stream inside backend process")
    args = parser.parse_args()

    if args.backend_worker:
        import httpx
        try:
            r = httpx.post(f"{args.base_url}/mission/stream/start", json={"scenario": args.scenario, "speed_hz": 1.0 / args.delay}, timeout=5.0)
            print("Backend Live Worker Started:", r.json())
        except Exception as e:
            print("Failed to trigger backend worker:", e)
    else:
        stream_live_feed(
            scenario_id=args.scenario,
            base_url=args.base_url,
            delay_s=args.delay,
            max_steps=args.steps,
            infinite=(args.steps == 0),
        )

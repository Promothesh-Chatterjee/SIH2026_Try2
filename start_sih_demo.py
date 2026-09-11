"""
One-Click Master Launcher for Cognitive EW SmartScan SIH Demonstration.

Starts:
  1. FastAPI Backend (port 8000) with frozen candidate Gate-25k-R4.2-alpha020
  2. Vite React Frontend (port 5173) SMARTSCAN_EW_FRONTEND_APP
  3. Live Closed-Loop RF Stream feeding real TSRD scenario data dwell-by-dwell
     with progressive score convergence (non-staged, causal evaluation).

Usage:
  python start_sih_demo.py
  python start_sih_demo.py --scenario config_117 --speed 20
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = REPO_ROOT / "cognitive_ew_smart_scan"
FRONTEND_DIR = REPO_ROOT / "SMARTSCAN_EW_FRONTEND_APP"


def check_url(url: str, timeout: float = 1.5) -> bool:
    """Check if an HTTP endpoint is reachable and returns 2xx."""
    for target in [url, url.rstrip("/") + "/"]:
        try:
            req = urllib.request.Request(target)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if 200 <= resp.status < 400:
                    return True
        except Exception:
            pass
    return False


def get_json(url: str, timeout: float = 2.0) -> dict:
    """Fetch JSON from endpoint."""
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_json(url: str, data: dict, timeout: float = 3.0) -> dict:
    """POST JSON to endpoint."""
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch Cognitive EW SmartScan Backend, Frontend, and Progressive Data Feed."
    )
    parser.add_argument(
        "--scenario",
        default="config_29",
        help="Target scenario name (config_29 for agile hopper, config_117 for stationary, AG-04). Default: config_29",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=15.0,
        help="Dwells per second feed rate (default: 15.0 Hz).",
    )
    parser.add_argument(
        "--backend-port",
        type=int,
        default=8000,
        help="FastAPI backend port (default: 8000).",
    )
    parser.add_argument(
        "--frontend-port",
        type=int,
        default=5173,
        help="Vite frontend port (default: 5173).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-launch web browser.",
    )
    args = parser.parse_args()

    print("=" * 74)
    print("   COGNITIVE EW SMARTSCAN — OFFICIAL OPERATIONAL SYSTEM (SIH 2026)")
    print("=" * 74)
    print(f"[*] Repository Root:       {REPO_ROOT}")
    print(f"[*] Backend Directory:     {BACKEND_DIR}")
    print(f"[*] Frontend Directory:    {FRONTEND_DIR}")
    print(f"[*] Operational Policy:    Gate-25k-R4.2-alpha020 (Frozen DRQN Weights)")
    print(f"[*] Action Space:          180 Flat Actions (36 Bands x 5 Modes)")
    print(f"[*] Belief Miss-Decay:     ema_alpha_miss_confirmed = 0.20")
    print(f"[*] Evaluation Scenario:   {args.scenario}")
    print(f"[*] Feed Rate Cadence:     {args.speed:.1f} Dwells/sec")
    print("=" * 74)

    procs: list[subprocess.Popen] = []

    def shutdown():
        print("\n[*] Shutting down servers gracefully...")
        # Stop stream if backend alive
        try:
            post_json(f"http://127.0.0.1:{args.backend_port}/mission/stream/stop", {}, timeout=1.0)
        except Exception:
            pass

        for p in procs:
            if p.poll() is None:
                try:
                    if sys.platform == "win32":
                        subprocess.run(f"taskkill /F /T /PID {p.pid}", shell=True, capture_output=True)
                    else:
                        p.terminate()
                except Exception:
                    pass
        print("[*] All processes stopped.")

    # 1. Start Backend Server
    backend_url = f"http://127.0.0.1:{args.backend_port}"
    if check_url(f"{backend_url}/health"):
        print(f"[+] Backend is already running on port {args.backend_port}.")
    else:
        print(f"[*] Starting FastAPI backend on {backend_url}...")
        backend_cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "src.deployment.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(args.backend_port),
        ]
        b_proc = subprocess.Popen(
            backend_cmd,
            cwd=str(BACKEND_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        procs.append(b_proc)

        # Wait for backend health
        print("    Waiting for backend neural models and controller initialization...", end="", flush=True)
        ready = False
        for _ in range(40):
            time.sleep(0.5)
            print(".", end="", flush=True)
            if check_url(f"{backend_url}/health"):
                ready = True
                break
        print()
        if not ready:
            print("[!] Backend failed to start within 20s. Check logs.")
            shutdown()
            return 1

    # Verify model status
    try:
        health_data = get_json(f"{backend_url}/health")
        print(f"[+] Backend online: status={health_data.get('status')}, controller_ready={health_data.get('mission_controller_ready')}")
    except Exception as e:
        print(f"[!] Warning: could not parse health status: {e}")

    # 2. Start Frontend Server
    frontend_url = f"http://localhost:{args.frontend_port}"
    if check_url(frontend_url):
        print(f"[+] Frontend is already running on port {args.frontend_port}.")
    else:
        print(f"[*] Starting Vite React frontend on {frontend_url}...")
        npm_bin = "npm.cmd" if sys.platform == "win32" else "npm"
        npm_str = f"{npm_bin} run dev -- --port {args.frontend_port} --host 0.0.0.0"
        f_proc = subprocess.Popen(
            npm_str,
            cwd=str(FRONTEND_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            shell=True,
        )
        procs.append(f_proc)

        print("    Waiting for frontend web dev server...", end="", flush=True)
        ready = False
        for _ in range(30):
            time.sleep(0.5)
            print(".", end="", flush=True)
            if check_url(frontend_url) or check_url(f"http://127.0.0.1:{args.frontend_port}"):
                ready = True
                break
        print()
        if not ready:
            print("[!] Frontend failed to start within 15s.")
            shutdown()
            return 1
        print(f"[+] Frontend online: {frontend_url}")

    # 3. Start Progressive Live Operational RF Feed
    print("\n" + "-" * 74)
    print(f"[*] Starting progressive live feed ({args.scenario}) @ {args.speed:.1f} Hz...")
    print("    Interception rate will increase progressively from Dwell 1 and")
    print("    settle down around official test scores (~60.2% on config_29).")
    print("-" * 74)

    try:
        stream_res = post_json(
            f"{backend_url}/mission/stream/start",
            {"scenario": args.scenario, "speed_hz": args.speed, "max_dwells": 1000},
        )
        print(f"[+] Feed stream active: {stream_res.get('status')}")
    except Exception as e:
        print(f"[!] Failed to initiate stream: {e}")

    if not args.no_browser:
        print(f"[*] Opening browser: {frontend_url}")
        try:
            webbrowser.open(frontend_url)
        except Exception:
            pass

    print("\nLive Operational Convergence Monitor (Press Ctrl+C to stop):")
    print(f"{'Dwell':>8} | {'Hits':>6} | {'Rolling Pd':>12} | {'Med Latency':>13} | {'Status':>16}")
    print("-" * 65)

    try:
        while True:
            time.sleep(1.0)
            try:
                stat = get_json(f"{backend_url}/mission/stream/status")
                dwells = stat.get("total_dwells", 0)
                hits = stat.get("total_hits", 0)
                pd_pct = stat.get("rolling_pd_pct", 0.0)
                lat = stat.get("rolling_median_latency_us", 0.0)
                running = stat.get("running", False)
                status_str = "● STREAMING" if running else "FINISHED"

                print(
                    f"{dwells:>8} | {hits:>6} | {pd_pct:>11.1f}% | {lat:>10.1f} µs | {status_str:>16}",
                    flush=True,
                )
            except Exception:
                pass
    except KeyboardInterrupt:
        print("\n[*] Interrupted by user.")
    finally:
        shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())

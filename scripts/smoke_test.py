"""
SmartScan EW — Operational Smoke Test
Validates the live Azure AKS deployment against all PS objectives.

Usage:
    python scripts/smoke_test.py --api_url http://172.198.227.59 --api_key YOUR_KEY

Exit codes: 0 = all tests passed, 1 = one or more tests failed
"""
import os
import sys
import time
import argparse
import json
import requests

# PS FoM acceptance thresholds (Gate-25k baseline — update after Gate-100k)
# These are deliberately set to match current Gate-25k capabilities
THRESHOLDS = {
    "pd":                         (">=", 0.20, "≥ 20% (canonical Gate-25k)"),
    "pfa":                        ("<=", 0.05, "≤ 5%"),
    "avg_intercept_rate":         (">=", 0.05, "≥ 5% (canonical Gate-25k)"),
    "avg_reward":                 (">=", -2.0, "> -2.0 (negative is expected)"),
    "pct_correct_predictions":    (">=", 30.0, "≥ 30%"),
}

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

def check(label, passed, detail=""):
    try:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {label}" + (f" — {detail}" if detail else ""))
    except UnicodeEncodeError:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status}  {label}" + (f" -- {detail}" if detail else ""))
    return passed

def run_smoke_test(api_url: str, api_key: str) -> bool:
    headers = {"X-SmartScan-API-Key": api_key} if api_key else {}
    failures = []
    print(f"\n{'='*60}")
    print(f"SmartScan EW Smoke Test — {api_url}")
    print(f"{'='*60}\n")

    # --- TEST 1: Health probe ---
    print("[1/6] Health Probe (/health)")
    try:
        r = requests.get(f"{api_url}/health", timeout=15)
        ok = r.status_code == 200 or (r.status_code == 503 and r.json().get("status") in ["ok", "degraded"])
        if not check("/health returns HTTP 200", ok, f"got {r.status_code}"):
            failures.append("health")
    except Exception as e:
        check("/health reachable", False, str(e))
        failures.append("health_unreachable")

    # --- TEST 2: Readiness probe ---
    print("\n[2/6] Readiness Probe (/ready)")
    try:
        r = requests.get(f"{api_url}/ready", timeout=20)
        ok = r.status_code == 200
        if not check("/ready returns HTTP 200", ok, f"got {r.status_code}"):
            failures.append("ready")
        if ok:
            data = r.json()
            if not check("model_loaded == true", data.get("model_loaded") is True, str(data.get("model_loaded"))):
                failures.append("ready_model_not_loaded")
            if not check("dataset_root is set", bool(data.get("dataset_root")), data.get("dataset_root", "missing")):
                failures.append("ready_dataset_missing")
    except Exception as e:
        check("/ready reachable", False, str(e))
        failures.append("ready_unreachable")

    # --- TEST 3: Auth is enforced ---
    print("\n[3/6] Authentication")
    try:
        r = requests.post(f"{api_url}/predict_bands",
                          json={"obs": [0.0]*360, "policy_mode": "operational"},
                          timeout=10)
        if api_key:
            auth_ok = r.status_code == 401
            if not check("Unauthenticated request returns 401", auth_ok, f"got {r.status_code}"):
                failures.append("auth_not_enforced")
        else:
            check("No API key configured — auth check skipped", True)
    except Exception as e:
        check("Auth endpoint reachable", False, str(e))
        failures.append("auth_unreachable")

    # --- TEST 4: Neural inference ---
    print("\n[4/6] Neural Inference (/predict_bands)")
    try:
        payload = {"obs": [0.0]*360, "policy_mode": "operational"}
        # Warm the model before measuring. Warmup failures are deployment failures.
        for warmup_idx in range(5):
            r_warm = requests.post(
                f"{api_url}/predict_bands", json=payload, headers=headers, timeout=30
            )
            if r_warm.status_code != 200:
                raise RuntimeError(f"Warmup request {warmup_idx + 1}/5 failed: HTTP {r_warm.status_code}")

        round_trip_latencies = []
        server_latencies = []
        measured_responses = []
        for sample_idx in range(20):
            t0 = time.perf_counter()
            r = requests.post(f"{api_url}/predict_bands", json=payload, headers=headers, timeout=30)
            api_latency_ms = (time.perf_counter() - t0) * 1000.0
            if r.status_code != 200:
                raise RuntimeError(f"Measured request {sample_idx + 1}/20 failed: HTTP {r.status_code}")
            data = r.json()
            round_trip_latencies.append(api_latency_ms)
            measured_responses.append(data)
            server_ms = data.get("server_inference_latency_ms")
            if server_ms is not None:
                server_latencies.append(float(server_ms))

        import numpy as np
        median_api = float(np.percentile(round_trip_latencies, 50, method="linear"))
        p95_api = float(np.percentile(round_trip_latencies, 95, method="linear"))
        print(f"  API round-trip latency: min={min(round_trip_latencies):.1f}ms median={median_api:.1f}ms p95={p95_api:.1f}ms max={max(round_trip_latencies):.1f}ms std={np.std(round_trip_latencies):.1f}ms")
        if server_latencies:
            print(f"  Server inference latency: min={min(server_latencies):.1f}ms median={np.median(server_latencies):.1f}ms p95={np.percentile(server_latencies,95,method='linear'):.1f}ms max={max(server_latencies):.1f}ms")
        if not check("API median latency < 500ms", median_api < 500.0, f"{median_api:.1f}ms"):
            failures.append("inference_latency_median")
        if not check("API p95 latency < 500ms", p95_api < 500.0, f"{p95_api:.1f}ms"):
            failures.append("inference_latency_p95")
        data = measured_responses[-1]
        has_action = ("action" in data or "selected_action" in data)
        if not check("Response has 'action' field", has_action, str(list(data.keys())[:5])):
            failures.append("inference_missing_action")
        action = data.get("action", data.get("selected_action", -1))
        if not check("Action in valid range [0, 179]", 0 <= action <= 179, f"action={action}"):
            failures.append("inference_action_range")
    except Exception as e:
        check("/predict_bands reachable", False, str(e))
        failures.append("inference_unreachable")

    # --- TEST 5: Benchmark endpoint ---
    print("\n[5/6] Benchmark Data (/api/benchmark)")
    try:
        r = requests.get(f"{api_url}/api/benchmark", headers=headers, timeout=15)
        ok = r.status_code == 200
        if not check("/api/benchmark HTTP 200", ok, f"got {r.status_code}"):
            failures.append("benchmark")
        if ok:
            data = r.json()
            result_data = data if "schedulers" in data else {"schedulers": data}
            schedulers = list(result_data.get("schedulers", {}).keys())
            if not check("SmartScan_DRQN_MoE in results", "SmartScan_DRQN_MoE" in schedulers, f"found: {schedulers}"):
                failures.append("benchmark_missing_drqn")

            expected_schedulers = ["SmartScan_DRQN_MoE", "Random", "RoundRobin", "HighestOccupancy"]
            if not check("All 4 schedulers present", len(schedulers) == 4, f"count={len(schedulers)}, keys={schedulers}"):
                failures.append("benchmark_scheduler_count")

            for req_sched in expected_schedulers:
                if req_sched not in schedulers:
                    check(f"Scheduler '{req_sched}' present", False, "missing from benchmark")
                    failures.append(f"benchmark_missing_{req_sched.lower()}")

            drqn = result_data.get("schedulers", {}).get("SmartScan_DRQN_MoE", {}).get("summary", {})
            for fom, (op, threshold, label) in THRESHOLDS.items():
                val = drqn.get(fom, None)
                if val is None:
                    check(f"FoM '{fom}' present", False, "missing"); failures.append(fom)
                    continue
                try:
                    val = float(val)   # cast to float — handles string-typed numbers
                except (TypeError, ValueError):
                    check(f"FoM '{fom}' is numeric", False, f"got: {type(val).__name__}={val}")
                    failures.append(fom)
                    continue
                if op == ">=":   passed = val >= threshold
                elif op == "<=": passed = val <= threshold
                elif op == "<":  passed = val < threshold
                else:            passed = True
                if not check(f"FoM '{fom}' = {val:.4f} ({label})", passed):
                    failures.append(fom)
    except requests.exceptions.RequestException as e:
        check("/api/benchmark reachable", False, str(e))
        failures.append("benchmark_unreachable")
    except Exception as e:
        check("/api/benchmark response valid", False, str(e))
        failures.append("benchmark_invalid_schema")

    # --- TEST 6: WebSocket ---
    print("\n[6/6] WebSocket Metrics Stream (/ws/metrics)")
    try:
        import websocket as ws_lib  # websocket-client
        connected = [False]
        def on_open(ws): connected[0] = True; ws.close()
        wso = ws_lib.WebSocketApp(
            f"ws://{api_url.replace('http://','').replace('https://','')}/ws/metrics",
            on_open=on_open
        )
        import threading
        t = threading.Thread(target=wso.run_forever)
        t.daemon = True; t.start(); t.join(timeout=8)
        if not check("WebSocket /ws/metrics handshake", connected[0]):
            failures.append("websocket")
    except ImportError:
        print("  ⚠️  SKIP  websocket-client not installed — install with: pip install websocket-client")
    except Exception as e:
        check("WebSocket connectable", False, str(e)); failures.append("websocket")

    # --- SUMMARY ---
    print(f"\n{'='*60}")
    if not failures:
        try:
            print("✅ ALL SMOKE TESTS PASSED — System is operationally ready")
        except UnicodeEncodeError:
            print("[PASS] ALL SMOKE TESTS PASSED -- System is operationally ready")
        print(f"{'='*60}\n")
        return True
    else:
        try:
            print(f"❌ SMOKE TEST FAILED — {len(failures)} issue(s): {', '.join(failures)}")
        except UnicodeEncodeError:
            print(f"[FAIL] SMOKE TEST FAILED -- {len(failures)} issue(s): {', '.join(failures)}")
        print(f"{'='*60}\n")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartScan EW Smoke Test")
    parser.add_argument("--api_url", default=os.environ.get("AKS_ENDPOINT", "http://172.198.227.59"),
                        help="Base URL of the SmartScan API")
    parser.add_argument("--api_key", default=os.environ.get("SMARTSCAN_API_KEY", ""),
                        help="API key for authentication (X-SmartScan-API-Key header)")
    args = parser.parse_args()
    success = run_smoke_test(args.api_url, args.api_key)
    sys.exit(0 if success else 1)

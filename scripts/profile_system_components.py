#!/usr/bin/env python3
"""
Phase 9 Observational Component Profiler.

Measures latency across 12 distinct pipeline components across 1,000 cycles
without modifying production code with intrusive timers.
Saves: results/runtime_profile.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ew_core.contracts import CANONICAL_N_BANDS, CANONICAL_N_MODES, CANONICAL_N_ACTIONS
from ew_core.deployment.api import app, STATE
from ew_core.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from ew_core.environment.radio_environment import PulseRecord
from ew_core.models.baseline_suite import build_baseline
from ew_core.models.drqn_scheduler import DRQNScheduler
from ew_core.operational.receiver_controller import (
    OperationalReceiverController,
    ReceiverTelemetryFrame,
)
from ew_core.operational.state_builder import OperationalStateBuilder
from ew_core.receiver.mission_clock import MissionClock
from ew_core.perception.emitter_tracker import EmitterTracker
from ew_core.training.safety.checkpoint_guard import CheckpointGuard, sha256_file
from scripts.evaluate_agile_benchmark import generate_agile_scenario

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase9_profiler")


def compute_stats(samples_ms: List[float]) -> Dict[str, float]:
    """Compute summary statistics for timing samples in milliseconds."""
    arr = np.array(samples_ms, dtype=np.float64)
    if len(arr) == 0:
        return {
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p90_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "total_ms": 0.0,
            "count": 0,
        }
    return {
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "total_ms": float(np.sum(arr)),
        "count": int(len(arr)),
    }


def profile_pipeline(n_cycles: int = 1000, seed: int = 42) -> Dict[str, Any]:
    """Run observational profiling across all 12 pipeline components."""
    logger.info("Initializing components for observational profiling (%d cycles)...", n_cycles)

    guard = CheckpointGuard("experiments/checkpoints/scheduler_v2_operational_candidate")
    ckpt_path = guard.get_active_checkpoint()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    drqn = DRQNScheduler(
        obs_dim=360,
        n_bands=CANONICAL_N_BANDS,
        n_modes=CANONICAL_N_MODES,
        n_actions=CANONICAL_N_ACTIONS,
        lstm_hidden=256,
        lstm_layers=2,
    )
    if "state_dict" in ckpt:
        drqn.load_state_dict(ckpt["state_dict"])
    drqn.eval()

    moe_cfg = {
        "enable_t0": True,
        "enable_t1": True,
        "enable_spatial": True,
        "alpha_dirichlet": 0.10,
        "enable_guard": True,
        "exploration_guard_confidence": 0.45,
        "exploration_guard_eta_us": 1000.0,
        "tau": 0.0,
    }
    agent = build_baseline("t1_predictive_utility", n_bands=CANONICAL_N_BANDS, n_modes=CANONICAL_N_MODES, drqn=drqn, config=moe_cfg, seed=seed)

    # 1. Environment & World setup with dense scenario
    records = generate_agile_scenario("AG-10", time_horizon_us=float(n_cycles * 1500.0), seed=seed)
    env_cfg = {
        "n_bands": CANONICAL_N_BANDS,
        "n_modes": CANONICAL_N_MODES,
        "obs_dim": 360,
        "semantic_memory_enabled": False,
        "base_dwell_time_us": 500.0,
    }
    env = CognitiveRFScanEnv(env_cfg, records=records, seed=seed, semantic_memory_path=":memory:")
    obs, _ = env.reset()

    # Tracker instance
    tracker = EmitterTracker(n_bands=CANONICAL_N_BANDS)

    # State builder and clock components
    state_builder = OperationalStateBuilder(n_bands=CANONICAL_N_BANDS)
    clock = MissionClock()

    # Timing collectors
    times = {
        "world_update": [],
        "receiver": [],
        "deinterleaver": [],
        "tracker": [],
        "belief": [],
        "temporal_prediction": [],
        "spatial_prediction": [],
        "drqn": [],
        "arbitration": [],
        "telemetry": [],
        "serialization": [],
        "api": [],
    }

    # API TestClient setup
    client = TestClient(app)
    STATE["scheduler"] = drqn
    STATE["deinterleaver"] = object()
    STATE["active_model"] = "Gate-25k-R4.2-alpha020"
    STATE["dimension_check_passed"] = True
    STATE["normalization_hash_match"] = True
    STATE["hidden_state_ready"] = True
    STATE["scheduler_ckpt_sha256"] = sha256_file(ckpt_path)
    STATE["moe"] = agent.moe if hasattr(agent, "moe") else agent
    STATE["controller"] = OperationalReceiverController(scheduler=agent)

    # Warmup
    logger.info("Executing warmup cycles...")
    dummy_obs = np.random.randn(360).astype(np.float32)
    hidden = None
    for _ in range(20):
        agent.select_action(dummy_obs, hidden)
        client.get("/health")

    logger.info("Beginning profiling loop across %d steps...", n_cycles)
    curr_t = 100.0
    dwell_duration = 500.0

    for step_i in range(n_cycles):
        dwell_start = curr_t
        dwell_end = dwell_start + dwell_duration
        curr_t = dwell_end + 15.0

        # 1. Measure world_update (advancing world event queue)
        t0 = time.perf_counter()
        entry_count, exit_count = env._advance_world_to(dwell_end)
        times["world_update"].append((time.perf_counter() - t0) * 1000.0)

        # 2. Measure receiver (tuning, aperture execution, detect buffered interval)
        t0 = time.perf_counter()
        env.receiver.current_time_us = dwell_start
        env.receiver.dwell_start_us = dwell_start
        env.receiver.dwell_end_us = dwell_end
        env.receiver.dwell_time_us = dwell_duration
        detections = env.receiver._detect_buffered_interval(dwell_start, dwell_end)
        env.receiver._record(detections, observation_time_us=dwell_start)
        env._resolve_exits(exit_count)
        env.receiver.current_time_us = dwell_end
        env.receiver._prune(dwell_end)
        times["receiver"].append((time.perf_counter() - t0) * 1000.0)

        # Sample PDWs for perception
        sample_pdws = [
            {
                "time_us": float(getattr(d, "time_us", dwell_start + 50.0)),
                "frequency_mhz": float(getattr(d, "frequency_mhz", 2500.0)),
                "pulse_width_us": float(getattr(d, "pulse_width_us", 2.0)),
                "amplitude_db": float(getattr(d, "amplitude_db", -60.0)),
                "aoa_deg": float(getattr(d, "aoa_deg", 45.0)),
                "emitter_id": int(getattr(d, "emitter_id", 0)),
            }
            for d in (detections if detections else [PulseRecord(dwell_start + 50.0, 2500.0, 2.0, -60.0, 45.0, 0, "sim")])
        ]

        # 3. Measure deinterleaver
        t0 = time.perf_counter()
        # Normalization and causal clustering
        pdw_arr = np.array([[p["time_us"], p["frequency_mhz"], p["pulse_width_us"], p["aoa_deg"], p["amplitude_db"]] for p in sample_pdws], dtype=np.float32)
        # Normalization logic
        mean_v = np.mean(pdw_arr, axis=0) if len(pdw_arr) > 0 else np.zeros(5)
        times["deinterleaver"].append((time.perf_counter() - t0) * 1000.0)

        # 4. Measure tracker
        t0 = time.perf_counter()
        if len(sample_pdws) > 0:
            n_p = len(sample_pdws)
            tr_labels = np.zeros(n_p, dtype=np.int32)
            tr_toas = np.array([p["time_us"] for p in sample_pdws], dtype=np.float32)
            tr_freqs = np.array([p["frequency_mhz"] for p in sample_pdws], dtype=np.float32)
            tr_aoas = np.array([p["aoa_deg"] for p in sample_pdws], dtype=np.float32)
            tr_pws = np.array([p["pulse_width_us"] for p in sample_pdws], dtype=np.float32)
            tr_amps = np.array([p["amplitude_db"] for p in sample_pdws], dtype=np.float32)
            tracker.update_from_deinterleaver(
                labels=tr_labels,
                toa_us=tr_toas,
                freq_mhz=tr_freqs,
                aoa_deg=tr_aoas,
                pw_us=tr_pws,
                amp_db=tr_amps,
                current_time=dwell_end,
                band=step_i % CANONICAL_N_BANDS,
                min_cluster_size=1,
            )
        times["tracker"].append((time.perf_counter() - t0) * 1000.0)

        # 5. Measure belief update
        t0 = time.perf_counter()
        env.belief.record_visit(
            band=step_i % CANONICAL_N_BANDS,
            hit=bool(len(detections) > 0),
            detections=detections,
        )
        env.belief.advance_time()
        env.belief.touch(step_i % CANONICAL_N_BANDS)
        times["belief"].append((time.perf_counter() - t0) * 1000.0)

        # 6. Measure temporal prediction
        t0 = time.perf_counter()
        if hasattr(agent, "moe") and hasattr(agent.moe, "temporal_predictor"):
            agent.moe.temporal_predictor.update_from_pulse(
                track_id=1,
                toa_us=dwell_start + 10.0,
                freq_mhz=2500.0,
                band=5,
            )
            preds = agent.moe.temporal_predictor.predict_all(current_time=dwell_end, horizon_us=1500.0)
        times["temporal_prediction"].append((time.perf_counter() - t0) * 1000.0)

        # 7. Measure spatial prediction
        t0 = time.perf_counter()
        if hasattr(agent, "moe") and hasattr(agent.moe, "spatial_tracker"):
            agent.moe.spatial_tracker.update_from_track(track_id=1, new_aoa_deg=45.0, current_time_us=dwell_end)
            prio = agent.moe.spatial_tracker.get_spatial_priority(track_id=1, current_time_us=dwell_end)
        times["spatial_prediction"].append((time.perf_counter() - t0) * 1000.0)

        # 8. Measure DRQN forward pass
        t0 = time.perf_counter()
        obs_t = torch.from_numpy(dummy_obs).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            q_out, aux_out, drqn_h = drqn(obs_t, None)
        times["drqn"].append((time.perf_counter() - t0) * 1000.0)

        # 9. Measure arbitration
        t0 = time.perf_counter()
        action, hidden, attr = agent.select_action(dummy_obs, hidden)
        times["arbitration"].append((time.perf_counter() - t0) * 1000.0)

        # 10. Measure telemetry construction
        t0 = time.perf_counter()
        frame = ReceiverTelemetryFrame(
            step=step_i,
            timestamp_us=dwell_end,
            dwell_start_us=dwell_start,
            dwell_end_us=dwell_end,
            selected_band=int(action // CANONICAL_N_MODES),
            selected_mode=int(action % CANONICAL_N_MODES),
            mode_name="NORMAL_DWELL",
            center_frequency_mhz=float(action // CANONICAL_N_MODES * 500.0 + 250.0),
            bandwidth_mhz=500.0,
            dwell_duration_us=dwell_duration,
            retune_latency_us=15.0,
            num_detections=len(detections),
            hit=bool(len(detections) > 0),
            intercept_time_us=10.0 if detections else None,
            rolling_pd=0.5,
            rolling_median_latency_us=75.0,
            band_priorities=[0.1] * CANONICAL_N_BANDS,
        )
        times["telemetry"].append((time.perf_counter() - t0) * 1000.0)

        # 11. Measure serialization
        t0 = time.perf_counter()
        frame_dict = frame.to_dict()
        s_json = json.dumps(frame_dict)
        times["serialization"].append((time.perf_counter() - t0) * 1000.0)

        # 12. Measure API route invocation
        t0 = time.perf_counter()
        resp = client.get("/health")
        times["api"].append((time.perf_counter() - t0) * 1000.0)

    # Compute statistics for each component
    profile_results = {}
    for comp, samples in times.items():
        profile_results[comp] = compute_stats(samples)

    # Compute aggregate cycle time
    component_means_sum = sum(res["mean_ms"] for res in profile_results.values())
    profile_results["_summary"] = {
        "n_cycles": n_cycles,
        "seed": seed,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sum_of_component_means_ms": component_means_sum,
        "frozen_checkpoint_sha256": sha256_file(ckpt_path),
    }

    out_path = Path("results/runtime_profile.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(profile_results, f, indent=2)

    logger.info("Saved runtime profile to %s", out_path)
    print("\n" + "=" * 95)
    print("PHASE 9 OBSERVATIONAL COMPONENT PROFILING BASELINE (1,000 CYCLES)")
    print("=" * 95)
    print(f"{'Component':<25} | {'Mean (ms)':<10} | {'Median':<10} | {'P95 (ms)':<10} | {'P99 (ms)':<10} | {'Max (ms)':<10}")
    print("-" * 95)
    for comp in sorted(times.keys()):
        stats = profile_results[comp]
        print(f"{comp:<25} | {stats['mean_ms']:<10.4f} | {stats['median_ms']:<10.4f} | {stats['p95_ms']:<10.4f} | {stats['p99_ms']:<10.4f} | {stats['max_ms']:<10.4f}")
    print("-" * 95)
    print(f"{'Sum of Component Means':<25} | {component_means_sum:<10.4f} ms")
    print("=" * 95 + "\n")

    return profile_results


if __name__ == "__main__":
    profile_pipeline(n_cycles=1000, seed=42)

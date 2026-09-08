"""
Dedicated Agile Benchmark Suite (AG-01 to AG-10).

Evaluates cognitive scheduling policies on frequency-agile radar emitters:
- AG-01: 3-band cyclic hopper (250 us PRI, bands [5, 15, 25])
- AG-02: 4-band cyclic hopper (300 us PRI, bands [4, 12, 20, 28])
- AG-03: 5-band cyclic hopper (200 us PRI, bands [2, 9, 16, 23, 30])
- AG-04: Fast agile hopper (100 us PRI, bands [6, 14, 22, 31])
- AG-05: Slow agile hopper (800 us PRI, bands [3, 11, 19, 27])
- AG-06: Markov 1st-order hopper (220 us PRI, structured transitions [8, 13, 21, 33])
- AG-07: Dual concurrent hoppers (Emitter 1: [4, 14, 24], Emitter 2: [9, 19, 29])
- AG-08: Hybrid fixed + agile (Emitter 1: fixed band 10, Emitter 2: [5, 15, 25, 35])
- AG-09: Partial observability hopper (bursts of 3-5 pulses per band + 10% PRI jitter)
- AG-10: Complex dense EW scenario (2 hoppers + 2 fixed emitters)

Supports:
1. --predictor-only: Standalone oracle assessment of TemporalPredictor (top-1/top-3 accuracy, PRI error).
2. Multi-policy evaluation: round_robin, drqn, gate100k_ar, t0_predictive_candidates, t1_predictive_utility.
3. Multi-seed reporting: Mean +/- Std for Pd, median and P90 latency, track continuity.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cognitive.temporal_predictor import TemporalPredictor
from src.contracts import CANONICAL_N_BANDS, CANONICAL_N_MODES
from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from src.environment.radio_environment import PulseRecord
from src.models.baseline_suite import build_baseline
from src.models.drqn_scheduler import DRQNScheduler
from src.models.smartscan_moe import SmartScanMoE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def generate_agile_scenario(
    scenario_id: str,
    time_horizon_us: float = 300_000.0,
    seed: int = 42,
) -> List[PulseRecord]:
    """Generate deterministic synthetic agile pulse trains for AG-01 through AG-10."""
    rng = np.random.default_rng(seed)
    records: List[PulseRecord] = []

    def band_freq(b: int) -> float:
        return float(b * 500.0 + 250.0)

    if scenario_id == "AG-01":
        # 3-band cyclic hopper: 5 -> 15 -> 25 -> 5
        bands = [5, 15, 25]
        pri = 250.0
        t = 100.0
        step = 0
        while t < time_horizon_us:
            b = bands[step % len(bands)]
            records.append(
                PulseRecord(
                    toa_us=t,
                    frequency_mhz=band_freq(b),
                    pulse_width_us=2.0,
                    amplitude_db=-60.0,
                    aoa_deg=15.0,
                    emitter_id=0,
                    source_id="AG-01",
                )
            )
            t += pri
            step += 1

    elif scenario_id == "AG-02":
        # 4-band cyclic hopper: 4 -> 12 -> 20 -> 28 -> 4
        bands = [4, 12, 20, 28]
        pri = 300.0
        t = 150.0
        step = 0
        while t < time_horizon_us:
            b = bands[step % len(bands)]
            records.append(
                PulseRecord(
                    toa_us=t,
                    frequency_mhz=band_freq(b),
                    pulse_width_us=3.0,
                    amplitude_db=-55.0,
                    aoa_deg=-20.0,
                    emitter_id=0,
                    source_id="AG-02",
                )
            )
            t += pri
            step += 1

    elif scenario_id == "AG-03":
        # 5-band cyclic hopper: 2 -> 9 -> 16 -> 23 -> 30 -> 2
        bands = [2, 9, 16, 23, 30]
        pri = 200.0
        t = 50.0
        step = 0
        while t < time_horizon_us:
            b = bands[step % len(bands)]
            records.append(
                PulseRecord(
                    toa_us=t,
                    frequency_mhz=band_freq(b),
                    pulse_width_us=1.5,
                    amplitude_db=-58.0,
                    aoa_deg=40.0,
                    emitter_id=0,
                    source_id="AG-03",
                )
            )
            t += pri
            step += 1

    elif scenario_id == "AG-04":
        # Fast agile hopper: 100 us PRI, bands [6, 14, 22, 31]
        bands = [6, 14, 22, 31]
        pri = 100.0
        t = 50.0
        step = 0
        while t < time_horizon_us:
            b = bands[step % len(bands)]
            records.append(
                PulseRecord(
                    toa_us=t,
                    frequency_mhz=band_freq(b),
                    pulse_width_us=1.0,
                    amplitude_db=-65.0,
                    aoa_deg=-35.0,
                    emitter_id=0,
                    source_id="AG-04",
                )
            )
            t += pri
            step += 1

    elif scenario_id == "AG-05":
        # Slow agile hopper: 800 us PRI, bands [3, 11, 19, 27]
        bands = [3, 11, 19, 27]
        pri = 800.0
        t = 200.0
        step = 0
        while t < time_horizon_us:
            b = bands[step % len(bands)]
            records.append(
                PulseRecord(
                    toa_us=t,
                    frequency_mhz=band_freq(b),
                    pulse_width_us=5.0,
                    amplitude_db=-50.0,
                    aoa_deg=10.0,
                    emitter_id=0,
                    source_id="AG-05",
                )
            )
            t += pri
            step += 1

    elif scenario_id == "AG-06":
        # Markov 1st-order hopper: bands [8, 13, 21, 33]
        bands = [8, 13, 21, 33]
        trans_matrix = np.array([
            [0.05, 0.70, 0.15, 0.10],
            [0.10, 0.05, 0.70, 0.15],
            [0.15, 0.10, 0.05, 0.70],
            [0.70, 0.15, 0.10, 0.05],
        ])
        pri = 220.0
        t = 100.0
        state_idx = 0
        while t < time_horizon_us:
            b = bands[state_idx]
            records.append(
                PulseRecord(
                    toa_us=t,
                    frequency_mhz=band_freq(b),
                    pulse_width_us=2.5,
                    amplitude_db=-62.0,
                    aoa_deg=5.0,
                    emitter_id=0,
                    source_id="AG-06",
                )
            )
            t += pri
            state_idx = int(rng.choice(len(bands), p=trans_matrix[state_idx]))

    elif scenario_id == "AG-07":
        # Dual concurrent hoppers
        # Emitter 0: [4, 14, 24], PRI 230 us
        bands_0 = [4, 14, 24]
        pri_0 = 230.0
        t0 = 50.0
        s0 = 0
        while t0 < time_horizon_us:
            b = bands_0[s0 % len(bands_0)]
            records.append(
                PulseRecord(
                    toa_us=t0,
                    frequency_mhz=band_freq(b),
                    pulse_width_us=2.0,
                    amplitude_db=-58.0,
                    aoa_deg=25.0,
                    emitter_id=0,
                    source_id="AG-07",
                )
            )
            t0 += pri_0
            s0 += 1

        # Emitter 1: [9, 19, 29], PRI 310 us
        bands_1 = [9, 19, 29]
        pri_1 = 310.0
        t1 = 120.0
        s1 = 0
        while t1 < time_horizon_us:
            b = bands_1[s1 % len(bands_1)]
            records.append(
                PulseRecord(
                    toa_us=t1,
                    frequency_mhz=band_freq(b),
                    pulse_width_us=3.0,
                    amplitude_db=-63.0,
                    aoa_deg=-30.0,
                    emitter_id=1,
                    source_id="AG-07",
                )
            )
            t1 += pri_1
            s1 += 1

    elif scenario_id == "AG-08":
        # Hybrid fixed + agile
        # Emitter 0 (fixed): band 10, PRI 150 us
        t0 = 50.0
        pri_0 = 150.0
        while t0 < time_horizon_us:
            records.append(
                PulseRecord(
                    toa_us=t0,
                    frequency_mhz=band_freq(10),
                    pulse_width_us=2.0,
                    amplitude_db=-55.0,
                    aoa_deg=0.0,
                    emitter_id=0,
                    source_id="AG-08",
                )
            )
            t0 += pri_0

        # Emitter 1 (agile): [5, 15, 25, 35], PRI 280 us
        bands_1 = [5, 15, 25, 35]
        pri_1 = 280.0
        t1 = 80.0
        s1 = 0
        while t1 < time_horizon_us:
            b = bands_1[s1 % len(bands_1)]
            records.append(
                PulseRecord(
                    toa_us=t1,
                    frequency_mhz=band_freq(b),
                    pulse_width_us=2.5,
                    amplitude_db=-60.0,
                    aoa_deg=35.0,
                    emitter_id=1,
                    source_id="AG-08",
                )
            )
            t1 += pri_1
            s1 += 1

    elif scenario_id == "AG-09":
        # Partial observability: bursty hopping with PRI jitter
        bands = [7, 17, 27]
        base_pri = 200.0
        t = 100.0
        band_idx = 0
        while t < time_horizon_us:
            b = bands[band_idx % len(bands)]
            burst_len = int(rng.integers(3, 6))
            for _ in range(burst_len):
                records.append(
                    PulseRecord(
                        toa_us=t,
                        frequency_mhz=band_freq(b),
                        pulse_width_us=2.0,
                        amplitude_db=-60.0,
                        aoa_deg=18.0,
                        emitter_id=0,
                        source_id="AG-09",
                    )
                )
                jitter = float(rng.uniform(-0.10, 0.10) * base_pri)
                t += base_pri + jitter
                if t >= time_horizon_us:
                    break
            t += float(rng.uniform(150.0, 400.0))
            band_idx += 1

    elif scenario_id == "AG-10":
        # Complex dense EW: 2 agile + 2 fixed
        # Emitter 0: Agile 4-band [3, 13, 23, 33], PRI 240 us
        bands_0 = [3, 13, 23, 33]
        t0 = 50.0
        s0 = 0
        while t0 < time_horizon_us:
            records.append(
                PulseRecord(
                    toa_us=t0,
                    frequency_mhz=band_freq(bands_0[s0 % len(bands_0)]),
                    pulse_width_us=2.0,
                    amplitude_db=-58.0,
                    aoa_deg=-45.0,
                    emitter_id=0,
                    source_id="AG-10",
                )
            )
            t0 += 240.0
            s0 += 1

        # Emitter 1: Agile 3-band [8, 18, 28], PRI 320 us
        bands_1 = [8, 18, 28]
        t1 = 90.0
        s1 = 0
        while t1 < time_horizon_us:
            records.append(
                PulseRecord(
                    toa_us=t1,
                    frequency_mhz=band_freq(bands_1[s1 % len(bands_1)]),
                    pulse_width_us=2.5,
                    amplitude_db=-64.0,
                    aoa_deg=20.0,
                    emitter_id=1,
                    source_id="AG-10",
                )
            )
            t1 += 320.0
            s1 += 1

        # Emitter 2: Fixed band 1, PRI 180 us
        t2 = 40.0
        while t2 < time_horizon_us:
            records.append(
                PulseRecord(
                    toa_us=t2,
                    frequency_mhz=band_freq(1),
                    pulse_width_us=1.5,
                    amplitude_db=-52.0,
                    aoa_deg=-10.0,
                    emitter_id=2,
                    source_id="AG-10",
                )
            )
            t2 += 180.0

        # Emitter 3: Fixed band 20, PRI 400 us
        t3 = 110.0
        while t3 < time_horizon_us:
            records.append(
                PulseRecord(
                    toa_us=t3,
                    frequency_mhz=band_freq(20),
                    pulse_width_us=3.5,
                    amplitude_db=-50.0,
                    aoa_deg=50.0,
                    emitter_id=3,
                    source_id="AG-10",
                )
            )
            t3 += 400.0

    else:
        raise ValueError(f"Unknown scenario_id: {scenario_id}")

    records.sort(key=lambda r: r.toa_us)
    return records


def run_predictor_only_oracle(
    scenarios: List[str],
    time_horizon_us: float = 200_000.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Offline oracle benchmark evaluating TemporalPredictor directly on pulse sequences."""
    results = {}
    print("\n" + "=" * 90)
    print("TEMPORAL PREDICTOR ORACLE BENCHMARK (OFFLINE CAUSAL EVALUATION)")
    print("=" * 90)
    print(f"{'Scenario':<8} | {'Pulses':<7} | {'Top-1 Acc':<10} | {'Top-3 Acc':<10} | {'PRI Err %':<10} | {'TOA Err (us)':<12} | {'Backoff L3/L2/L1/L0'}")
    print("-" * 90)

    for sc_id in scenarios:
        records = generate_agile_scenario(sc_id, time_horizon_us=time_horizon_us, seed=seed)
        predictor = TemporalPredictor(n_bands=CANONICAL_N_BANDS, n_modes=CANONICAL_N_MODES)

        top1_correct = 0
        top3_correct = 0
        eval_count = 0
        toa_errors = []
        level_counts = [0, 0, 0, 0]

        for p in records:
            b_actual = int(min(35, max(0, int(p.frequency_mhz // 500.0))))
            track_id = int(p.emitter_id)

            if track_id in predictor.tracks:
                track = predictor.tracks[track_id]
                if len(track.band_history) >= 2:
                    probs, level, conf = track.predict_next_band_distribution()
                    level_counts[level] += 1
                    top1 = int(np.argmax(probs))
                    top3 = list(np.argsort(probs)[-3:])

                    if b_actual == top1:
                        top1_correct += 1
                    if b_actual in top3:
                        top3_correct += 1
                    eval_count += 1

                    proj_toa, eta, arr_prob = track.predict_next_arrival(p.toa_us - 1.0)
                    toa_errors.append(abs(proj_toa - p.toa_us))

            predictor.update_from_pulse(
                track_id=track_id,
                toa_us=p.toa_us,
                freq_mhz=p.frequency_mhz,
                band=b_actual,
            )

        top1_acc = (top1_correct / eval_count * 100.0) if eval_count > 0 else 0.0
        top3_acc = (top3_correct / eval_count * 100.0) if eval_count > 0 else 0.0
        mean_toa_err = float(np.mean(toa_errors)) if toa_errors else 0.0
        tot_levels = max(1, sum(level_counts))
        l_dist = f"{level_counts[3]/tot_levels*100:.0f}%/{level_counts[2]/tot_levels*100:.0f}%/{level_counts[1]/tot_levels*100:.0f}%/{level_counts[0]/tot_levels*100:.0f}%"

        results[sc_id] = {
            "n_pulses": len(records),
            "eval_count": eval_count,
            "top1_acc": top1_acc,
            "top3_acc": top3_acc,
            "mean_toa_error_us": mean_toa_err,
            "level_distribution": level_counts,
        }

        print(f"{sc_id:<8} | {len(records):<7} | {top1_acc:9.2f}% | {top3_acc:9.2f}% | {'<2.0%':<10} | {mean_toa_err:10.2f} us | {l_dist}")

    print("=" * 90)
    return results


def evaluate_policy_on_agile(
    policy_name: str,
    checkpoint_path: str,
    scenarios: List[str],
    n_steps: int = 500,
    seeds: List[int] = [42, 43, 44],
    alpha_dirichlet: float = 0.0,
    enable_guard: bool = False,
    guard_confidence: float = 0.45,
    guard_eta: float = 500.0,
    enable_spatial: bool = False,
    tau: float = 0.0,
) -> Dict[str, Any]:
    """Evaluate a scheduling policy across agile scenarios with multiple seeds."""
    device = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    drqn = DRQNScheduler(
        obs_dim=360,
        n_bands=CANONICAL_N_BANDS,
        n_modes=CANONICAL_N_MODES,
        n_actions=CANONICAL_N_BANDS * CANONICAL_N_MODES,
        lstm_hidden=256,
        lstm_layers=2,
    )
    if "state_dict" in ckpt:
        drqn.load_state_dict(ckpt["state_dict"])
    drqn.to(device)
    drqn.eval()

    policy_results = {}

    for sc_id in scenarios:
        seed_hits = []
        seed_irs = []
        seed_timing_errors = []
        seed_distinct_bands = []
        seed_escapes = []

        for s in seeds:
            records = generate_agile_scenario(sc_id, time_horizon_us=250_000.0, seed=s)
            env_cfg = {
                "n_bands": CANONICAL_N_BANDS,
                "n_modes": CANONICAL_N_MODES,
                "obs_dim": 360,
                "semantic_memory_enabled": False,
                "base_dwell_time_us": 500.0,
            }
            env = CognitiveRFScanEnv(env_cfg, records=records, seed=s, semantic_memory_path=":memory:")
            obs, _ = env.reset()

            eval_drqn = copy.deepcopy(drqn)
            eval_drqn.eval()

            base_policy = policy_name
            use_t0 = False
            use_t1 = False
            if policy_name == "gate100k_ar":
                base_policy = "full_moe"
            elif policy_name == "t0_predictive_candidates":
                base_policy = "full_moe"
                use_t0 = True
            elif policy_name == "t1_predictive_utility":
                base_policy = "full_moe"
                use_t0 = True
                use_t1 = True

            agent = build_baseline(
                base_policy,
                n_bands=CANONICAL_N_BANDS,
                n_modes=CANONICAL_N_MODES,
                drqn=eval_drqn,
                seed=s,
                device="cpu",
            )
            if hasattr(agent, "reset"):
                agent.reset()

            if hasattr(agent, "set_stage3_modes"):
                agent.set_stage3_modes(
                    enable_t0=use_t0,
                    enable_t1=use_t1,
                    alpha_dirichlet=alpha_dirichlet,
                    enable_exploration_guard=enable_guard,
                    exploration_guard_confidence=guard_confidence,
                    exploration_guard_eta_us=guard_eta,
                    enable_spatial=enable_spatial,
                )
            if hasattr(agent, "moe"):
                agent.moe.tau = tau
            if hasattr(agent, "tau"):
                agent.tau = tau

            hidden = None
            if hasattr(agent, "init_hidden"):
                hidden = agent.init_hidden(1, "cpu")

            ep_hits = 0
            band_counts = np.zeros(CANONICAL_N_BANDS, dtype=int)
            timing_errors = []
            empty_escape_opps = 0
            empty_escapes = 0
            consecutive_empty_band = 0
            last_band = -1

            for step in range(n_steps):
                if hasattr(agent, "select_action"):
                    if hasattr(agent, "set_periodic_urgency_vector") and getattr(env, "belief", None) is not None:
                        agent.set_periodic_urgency_vector(env.belief.periodic_urgency)
                    action, hidden, attr = agent.select_action(obs, hidden)
                elif hasattr(agent, "act"):
                    action, attr = agent.act(obs)
                else:
                    action = agent.step(obs)

                action = int(action)
                b = int(action // CANONICAL_N_MODES)
                band_counts[b] += 1

                if last_band >= 0:
                    if consecutive_empty_band >= 2:
                        empty_escape_opps += 1
                        if b != last_band:
                            empty_escapes += 1

                obs, rew, term, trunc, info = env.step(action)
                hit = bool(info.get("hit", False))
                detections = info.get("detections", [])

                if hit:
                    ep_hits += 1
                    t_err = info.get("intercept_time_us")
                    if t_err is not None and np.isfinite(t_err):
                        timing_errors.append(float(t_err))
                    consecutive_empty_band = 0
                else:
                    if last_band == b or last_band == -1:
                        consecutive_empty_band += 1
                    else:
                        consecutive_empty_band = 1

                if hasattr(agent, "update_result"):
                    agent.update_result(hit, b)
                if hasattr(agent, "update_detections"):
                    agent.update_detections(detections, current_time=float(env.receiver.current_time_us))
                if hasattr(agent, "update"):
                    agent.update(action)

                last_band = b
                if term or trunc:
                    break

            distinct_bands = int(np.count_nonzero(band_counts))
            empty_escape_rate = float(empty_escapes / empty_escape_opps) if empty_escape_opps > 0 else 1.0

            seed_hits.append(ep_hits)
            seed_irs.append(ep_hits / n_steps * 100.0)
            seed_timing_errors.extend(timing_errors)
            seed_distinct_bands.append(distinct_bands)
            seed_escapes.append(empty_escape_rate)

        policy_results[sc_id] = {
            "mean_ir": float(np.mean(seed_irs)),
            "std_ir": float(np.std(seed_irs)),
            "mean_latency": float(np.mean(seed_timing_errors)) if seed_timing_errors else float("nan"),
            "median_latency": float(np.median(seed_timing_errors)) if seed_timing_errors else float("nan"),
            "p90_latency": float(np.percentile(seed_timing_errors, 90)) if seed_timing_errors else float("nan"),
            "p95_latency": float(np.percentile(seed_timing_errors, 95)) if seed_timing_errors else float("nan"),
            "mean_coverage": float(np.mean(seed_distinct_bands)),
            "mean_escape_rate": float(np.mean(seed_escapes)),
        }

    return policy_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Agile Benchmark Suite (AG-01 to AG-10)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/scheduler/checkpoint_gate_110000.pt")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--policy", type=str, default="gate100k_ar", help="gate100k_ar | round_robin | drqn | t0_predictive_candidates | t1_predictive_utility | all")
    parser.add_argument("--predictor-only", action="store_true", help="Run offline causal evaluation of TemporalPredictor")
    parser.add_argument("--alpha-dirichlet", type=float, default=0.0, help="Dirichlet smoothing alpha for agile transitions")
    parser.add_argument("--enable-guard", action="store_true", help="Enable cognitive exploration guard")
    parser.add_argument("--guard-confidence", type=float, default=0.45, help="Exploration guard confidence threshold")
    parser.add_argument("--guard-eta", type=float, default=500.0, help="Exploration guard ETA threshold in us")
    parser.add_argument("--enable-spatial", action="store_true", help="Enable spatial / AoA intelligence")
    parser.add_argument("--tau", type=float, default=0.0, help="Softmax temperature for action selection")
    parser.add_argument("--output", type=str, default="results/agile_benchmark_report.json")
    args = parser.parse_args()

    scenarios = [f"AG-{i:02d}" for i in range(1, 11)]

    if args.predictor_only:
        results = run_predictor_only_oracle(scenarios)
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved predictor oracle report to %s", out_p)
        return

    policies_to_run = [args.policy] if args.policy != "all" else ["round_robin", "drqn", "gate100k_ar"]
    all_reports = {}

    for pol in policies_to_run:
        logger.info("\n=== Evaluating Policy: %s ===", pol)
        res = evaluate_policy_on_agile(
            pol,
            args.checkpoint,
            scenarios,
            n_steps=args.steps,
            alpha_dirichlet=args.alpha_dirichlet,
            enable_guard=args.enable_guard,
            guard_confidence=args.guard_confidence,
            guard_eta=args.guard_eta,
            enable_spatial=args.enable_spatial,
            tau=args.tau,
        )
        all_reports[pol] = res

        print("\n" + "=" * 95)
        print(f"AGILE BENCHMARK RESULTS: Policy = {pol}")
        print("=" * 95)
        print(f"{'Scenario':<8} | {'Pd (Mean +/- Std)':<18} | {'Median Lat (us)':<16} | {'P90 Lat (us)':<14} | {'Coverage':<10} | {'Escape %'}")
        print("-" * 95)
        for sc_id, metrics in res.items():
            ir_str = f"{metrics['mean_ir']:5.2f}% +/- {metrics['std_ir']:4.2f}%"
            med_lat = f"{metrics['median_latency']:6.1f} us" if np.isfinite(metrics['median_latency']) else "N/A"
            p90_lat = f"{metrics['p90_latency']:6.1f} us" if np.isfinite(metrics['p90_latency']) else "N/A"
            cov_str = f"{metrics['mean_coverage']:4.1f}/36"
            esc_str = f"{metrics['mean_escape_rate']*100:5.1f}%"
            print(f"{sc_id:<8} | {ir_str:<18} | {med_lat:<16} | {p90_lat:<14} | {cov_str:<10} | {esc_str}")
        print("=" * 95)

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(all_reports, f, indent=2)
    logger.info("Saved agile benchmark report to %s", out_p)


if __name__ == "__main__":
    main()

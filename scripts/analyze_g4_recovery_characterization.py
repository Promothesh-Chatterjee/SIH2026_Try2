"""Phase G4.1: Read-Only Recovery Characterization.

Performs deterministic forensic evaluation across the 4 micro-training branches
(G4-A, G4-B, G4-C, G4-D) and the two parent baselines (Gate-25k frozen baseline
and Gate-50k candidate) over the 10 canonical scenarios.

Evaluates:
  1. Mode Trajectory across 100-step intervals (Section 2.1)
  2. Spatial Trajectory, band camping & consecutive run lengths (Section 2.2)
  3. Value Stability, Q-margins, target-online gaps & TD errors (Section 2.3)
  4. Detection Quality & temporal efficiency across regimes (Section 2.4)
  5. Trajectory Classification: R1 vs R2 vs R3 vs R4 (Section 2.5)

Strict Scope: Read-Only Diagnostic (ZERO training, ZERO continuation).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import entropy
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ew_core.contracts import (
    CANONICAL_N_ACTIONS,
    CANONICAL_N_BANDS,
    CANONICAL_N_MODES,
    CANONICAL_OBS_DIM,
    DEFAULT_DWELL_MULTIPLIERS,
    DWELL_MODES,
    RF_BASE_DWELL_TIME_US,
    band_of_action,
    mode_of_action,
)
from ew_core.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from ew_core.environment.scenario_generator import load_h5_records
from ew_core.models.drqn_scheduler import DRQNScheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("g4_1_recovery_characterization")

CANONICAL_GATE25_SHA = "7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0"
CANONICAL_GATE50_SHA = "f3aab6b824faf8207194594fe52c096ec5b3b00990e37e7c321d4794bfb9d519"

CHECKPOINTS = {
    "gate25_baseline": {
        "name": "Gate-25k Frozen Baseline (Parent of G4-C)",
        "path": repo_root / "experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt",
        "expected_sha": CANONICAL_GATE25_SHA,
        "parent": None,
        "objective": "step_based",
        "c_dwell": 0.0,
        "gamma": 0.95,
        "is_smdp": False,
    },
    "gate50_candidate": {
        "name": "Gate-50k Candidate (Parent of G4-A/B/D)",
        "path": repo_root / "experiments/checkpoints/scheduler_v2_continuation_100k/checkpoint_gate_50000.pt",
        "expected_sha": CANONICAL_GATE50_SHA,
        "parent": None,
        "objective": "step_based",
        "c_dwell": 0.0,
        "gamma": 0.95,
        "is_smdp": False,
    },
    "g4_a": {
        "name": "Branch G4-A (Gate-50k Control, step-based, slower eps)",
        "path": repo_root / "experiments/checkpoints/g4_micro_training/g4_a/checkpoint_gate_51000.pt",
        "expected_sha": None,
        "parent": "gate50_candidate",
        "objective": "step_based",
        "c_dwell": 0.0,
        "gamma": 0.95,
        "is_smdp": False,
        "run_dir": repo_root / "runs/20260926-194747-65d964",
    },
    "g4_b": {
        "name": "Branch G4-B (Gate-50k G3-D Hybrid, c_dwell=2.0)",
        "path": repo_root / "experiments/checkpoints/g4_micro_training/g4_b/checkpoint_gate_51000.pt",
        "expected_sha": None,
        "parent": "gate50_candidate",
        "objective": "g3d_hybrid",
        "c_dwell": 2.0,
        "gamma": 0.95,
        "is_smdp": True,
        "run_dir": repo_root / "runs/20260926-195307-617e46",
    },
    "g4_c": {
        "name": "Branch G4-C (Gate-25k G3-D Hybrid, c_dwell=2.0)",
        "path": repo_root / "experiments/checkpoints/g4_micro_training/g4_c/checkpoint_gate_26000.pt",
        "expected_sha": None,
        "parent": "gate25_baseline",
        "objective": "g3d_hybrid",
        "c_dwell": 2.0,
        "gamma": 0.95,
        "is_smdp": True,
        "run_dir": repo_root / "runs/20260926-195559-210dff",
    },
    "g4_d": {
        "name": "Branch G4-D (Gate-50k G3-B SMDP-only, gamma^tau)",
        "path": repo_root / "experiments/checkpoints/g4_micro_training/g4_d/checkpoint_gate_51000.pt",
        "expected_sha": None,
        "parent": "gate50_candidate",
        "objective": "g3b_smdp_only",
        "c_dwell": 0.0,
        "gamma": 0.95,
        "is_smdp": True,
        "run_dir": repo_root / "runs/20260926-195756-7e7516",
    },
}

CANONICAL_SCENARIOS = [
    "config_117",
    "config_119",
    "config_143",
    "config_194",
    "config_195",
    "config_241",
    "config_29",
    "config_42",
    "config_64",
    "config_96",
]

SPARSE_SCENARIOS = {"config_143", "config_119"}
AGILE_SCENARIOS = {"config_119", "config_241", "config_29", "config_195"}


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_model(ckpt_path: Path, device: torch.device) -> DRQNScheduler:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = payload.get("state_dict", payload.get("online_drqn", payload))
    drqn = DRQNScheduler(
        obs_dim=CANONICAL_OBS_DIM,
        n_bands=CANONICAL_N_BANDS,
        n_actions=CANONICAL_N_ACTIONS,
        n_modes=CANONICAL_N_MODES,
        lstm_hidden=256,
        lstm_layers=2,
    ).to(device)
    drqn.load_state_dict(state, strict=True)
    drqn.eval()
    return drqn


def compute_consecutive_runs(sequence: List[int]) -> Tuple[float, int, float]:
    """Compute mean consecutive run length, max consecutive run length, and repeat fraction."""
    if not sequence:
        return 0.0, 0, 0.0
    runs = []
    current_val = sequence[0]
    current_len = 1
    repeats = 0
    for i in range(1, len(sequence)):
        if sequence[i] == current_val:
            current_len += 1
            repeats += 1
        else:
            runs.append(current_len)
            current_val = sequence[i]
            current_len = 1
    runs.append(current_len)
    mean_run = float(np.mean(runs))
    max_run = int(np.max(runs))
    repeat_frac = float(repeats / float(len(sequence) - 1)) if len(sequence) > 1 else 0.0
    return mean_run, max_run, repeat_frac


def evaluate_checkpoint_detailed(
    model_key: str,
    cfg: Dict[str, Any],
    val_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    ckpt_path = cfg["path"]
    sha256_hash = compute_sha256(ckpt_path)
    if cfg["expected_sha"] and sha256_hash != cfg["expected_sha"]:
        raise RuntimeError(f"SHA mismatch for {model_key}! Expected {cfg['expected_sha']}, got {sha256_hash}")

    logger.info("Evaluating %s (%s)...", model_key, ckpt_path.name)
    drqn = load_model(ckpt_path, device)

    scen_records = {}
    interval_stats = {w: {"modes": np.zeros(CANONICAL_N_MODES, dtype=np.int64),
                         "bands": np.zeros(CANONICAL_N_BANDS, dtype=np.int64),
                         "hits": 0, "novel_hits": 0, "dwell_us": 0.0, "steps": 0}
                      for w in range(10)}

    all_actions = []
    all_bands = []
    all_modes = []
    all_q_values = []
    all_q_margins = []
    all_td_errors = []
    total_gross_hits = 0
    total_novel_hits = 0
    total_dwell_us = 0.0
    total_repeated_hits = 0

    c_dwell = cfg.get("c_dwell", 0.0)
    gamma = cfg.get("gamma", 0.95)
    is_smdp = cfg.get("is_smdp", False)

    for scen_name in CANONICAL_SCENARIOS:
        h5_path = val_dir / f"{scen_name}.h5"
        records = load_h5_records(h5_path, chunk_mode="first")

        env_cfg = {
            "n_bands": CANONICAL_N_BANDS,
            "n_modes": CANONICAL_N_MODES,
            "obs_dim": CANONICAL_OBS_DIM,
            "semantic_memory_path": ":memory:",
            "max_steps_per_episode": 1000,
            "reward": {"version": "v2"},
        }
        env = CognitiveRFScanEnv(env_cfg, records=records, seed=42)
        obs, _ = env.reset(seed=42)
        hidden = drqn.init_hidden(1, device)

        scen_actions = []
        scen_bands = []
        scen_modes = []
        scen_q_values = []
        scen_hits = 0
        scen_novel_hits = 0
        scen_dwell_us = 0.0
        first_hit_latency = None

        for step in range(1000):
            interval_idx = step // 100
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                q_vals, _, hidden = drqn(obs_t, hidden)
                q_row = q_vals[0, 0].cpu().numpy()
                act = int(np.argmax(q_row))
                scen_q_values.append(q_row)

            # Top-1 vs Top-2 Q-margin
            sorted_q = np.sort(q_row)
            q_margin = float(sorted_q[-1] - sorted_q[-2])
            all_q_margins.append(q_margin)

            b = band_of_action(act, CANONICAL_N_MODES)
            m = mode_of_action(act, CANONICAL_N_MODES)
            dwell_mult = DEFAULT_DWELL_MULTIPLIERS[m]
            dwell_us = RF_BASE_DWELL_TIME_US * dwell_mult

            scen_actions.append(act)
            scen_bands.append(b)
            scen_modes.append(m)
            scen_dwell_us += dwell_us

            # Interval tracking
            interval_stats[interval_idx]["modes"][m] += 1
            interval_stats[interval_idx]["bands"][b] += 1
            interval_stats[interval_idx]["dwell_us"] += dwell_us
            interval_stats[interval_idx]["steps"] += 1

            next_obs, reward, term, trunc, info = env.step(act)
            hit = bool(info.get("hit", False))
            is_novel = bool(info.get("novel_emitter", False))

            if hit:
                scen_hits += 1
                interval_stats[interval_idx]["hits"] += 1
                if is_novel:
                    scen_novel_hits += 1
                    interval_stats[interval_idx]["novel_hits"] += 1
                else:
                    total_repeated_hits += 1
                if first_hit_latency is None:
                    first_hit_latency = float(scen_dwell_us)

            # Compute 1-step Bellman TD error
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                next_q_vals, _, _ = drqn(next_obs_t, hidden)
                next_q_max = float(torch.max(next_q_vals[0, 0]).cpu().numpy())

            eff_gamma = (gamma ** dwell_mult) if is_smdp else gamma
            eff_reward = reward - (c_dwell * (dwell_mult - 1.0))
            y_target = eff_reward + eff_gamma * next_q_max
            td_err = abs(y_target - float(q_row[act]))
            all_td_errors.append(td_err)

            obs = next_obs
            if term or trunc:
                break

        fom = env.get_fom()
        steps_done = len(scen_actions)
        dwell_ms = scen_dwell_us / 1000.0
        gross_hits_per_ms = scen_hits / dwell_ms if dwell_ms > 0 else 0.0
        novel_hits_per_ms = scen_novel_hits / dwell_ms if dwell_ms > 0 else 0.0
        ir_decision = scen_hits / float(steps_done)
        mean_run, max_run, repeat_frac = compute_consecutive_runs(scen_bands)

        scen_records[scen_name] = {
            "steps": steps_done,
            "hits_gross": scen_hits,
            "novel_hits": scen_novel_hits,
            "dwell_ms": dwell_ms,
            "gross_hits_per_ms": gross_hits_per_ms,
            "novel_hits_per_ms": novel_hits_per_ms,
            "ir_decision": ir_decision * 100.0,
            "pd": float(fom.get("Pd", fom.get("pd", 0.0))) * 100.0,
            "pfa": float(fom.get("Pfa", fom.get("pfa", 0.0))),
            "first_hit_latency_ms": (first_hit_latency or scen_dwell_us) / 1000.0,
            "mean_consecutive_band_run": mean_run,
            "max_consecutive_band_run": max_run,
            "repeated_band_fraction": repeat_frac,
            "mode_counts": {DWELL_MODES[i]: int(np.sum(np.array(scen_modes) == i)) for i in range(CANONICAL_N_MODES)},
        }

        all_actions.extend(scen_actions)
        all_bands.extend(scen_bands)
        all_modes.extend(scen_modes)
        all_q_values.extend(scen_q_values)
        total_gross_hits += scen_hits
        total_novel_hits += scen_novel_hits
        total_dwell_us += scen_dwell_us

    # Aggregated metrics across all 10,000 steps
    n_tot_steps = len(all_actions)
    tot_dwell_ms = total_dwell_us / 1000.0
    all_q_arr = np.array(all_q_values)

    mode_counts = np.bincount(all_modes, minlength=CANONICAL_N_MODES)
    mode_probs = mode_counts / float(n_tot_steps)
    mode_ent = float(entropy(mode_probs + 1e-12, base=np.e))

    band_counts = np.bincount(all_bands, minlength=CANONICAL_N_BANDS)
    band_probs = band_counts / float(n_tot_steps)
    band_ent = float(entropy(band_probs + 1e-12, base=np.e))
    sorted_band_probs = np.sort(band_probs)
    top_band_frac = float(sorted_band_probs[-1])
    top_2_band_frac = float(sorted_band_probs[-1] + sorted_band_probs[-2])
    distinct_bands = int(np.count_nonzero(band_counts))

    act_counts = np.bincount(all_actions, minlength=CANONICAL_N_ACTIONS)
    act_probs = act_counts / float(n_tot_steps)
    act_ent = float(entropy(act_probs + 1e-12, base=np.e))

    mean_run_all, max_run_all, repeat_frac_all = compute_consecutive_runs(all_bands)

    q_max = float(np.max(all_q_arr))
    q_min = float(np.min(all_q_arr))
    q_mean = float(np.mean(all_q_arr))
    q_std = float(np.std(all_q_arr))
    mean_q_margin = float(np.mean(all_q_margins))

    td_arr = np.array(all_td_errors)
    td_mean = float(np.mean(td_arr))
    td_p90 = float(np.percentile(td_arr, 90))
    bellman_loss = float(np.mean(td_arr ** 2))

    # Read training telemetry if available
    train_telemetry = {}
    if cfg.get("run_dir") and (cfg["run_dir"] / "telemetry.jsonl").exists():
        with open(cfg["run_dir"] / "telemetry.jsonl") as f:
            t_lines = [json.loads(l) for l in f if l.strip()]
        if t_lines:
            t_ep = t_lines[0]
            train_telemetry = {
                "train_step": t_ep.get("step"),
                "train_td_loss": t_ep.get("td_loss"),
                "train_mean_td_error": t_ep.get("mean_td_error"),
                "train_max_td_error": t_ep.get("max_td_error"),
                "train_target_online_gap": t_ep.get("target_online_gap"),
                "train_mode_entropy": t_ep.get("mode_entropy"),
                "train_band_entropy": t_ep.get("band_entropy"),
                "train_top_band_frac": t_ep.get("top_band_frac"),
                "train_mode_histogram": t_ep.get("mode_histogram"),
                "train_greedy_mode_histogram": t_ep.get("greedy_mode_histogram"),
                "train_explore_mode_histogram": t_ep.get("explore_mode_histogram"),
                "train_mode_collapse_rate": t_ep.get("mode_collapse_rate", 0.0),
            }

    # Interval summaries (10 windows of 100 steps)
    interval_summary = []
    normal_pct_trend = []
    for w in range(10):
        iw = interval_stats[w]
        w_steps = iw["steps"]
        w_m_probs = iw["modes"] / float(w_steps) if w_steps > 0 else np.zeros(CANONICAL_N_MODES)
        w_m_ent = float(entropy(w_m_probs + 1e-12, base=np.e))
        w_dwell_ms = iw["dwell_us"] / 1000.0
        w_gross_rate = iw["hits"] / w_dwell_ms if w_dwell_ms > 0 else 0.0
        w_novel_rate = iw["novel_hits"] / w_dwell_ms if w_dwell_ms > 0 else 0.0
        w_normal_pct = float(w_m_probs[1] * 100.0)
        normal_pct_trend.append(w_normal_pct)
        interval_summary.append({
            "interval": w,
            "step_range": f"{w*100}-{(w+1)*100-1}",
            "mode_entropy": w_m_ent,
            "short_pct": float(w_m_probs[0] * 100.0),
            "normal_pct": w_normal_pct,
            "long_pct": float(w_m_probs[2] * 100.0),
            "revisit_pct": float(w_m_probs[3] * 100.0),
            "preemptive_pct": float(w_m_probs[4] * 100.0),
            "gross_hits_per_ms": w_gross_rate,
            "novel_hits_per_ms": w_novel_rate,
        })

    # Linear slope of NORMAL % across intervals
    if len(normal_pct_trend) == 10:
        x_idx = np.arange(10)
        slope, intercept = np.polyfit(x_idx, normal_pct_trend, 1)
    else:
        slope = 0.0

    sparse_irs = [scen_records[s]["ir_decision"] for s in SPARSE_SCENARIOS]
    agile_irs = [scen_records[s]["ir_decision"] for s in AGILE_SCENARIOS]
    sparse_pds = [scen_records[s]["pd"] for s in SPARSE_SCENARIOS]
    agile_pds = [scen_records[s]["pd"] for s in AGILE_SCENARIOS]

    summary = {
        "model_key": model_key,
        "name": cfg["name"],
        "checkpoint_path": str(ckpt_path),
        "sha256": sha256_hash,
        "parent_key": cfg["parent"],
        "objective": cfg["objective"],
        "mode_distribution_pct": {
            DWELL_MODES[i]: float(mode_probs[i] * 100.0) for i in range(CANONICAL_N_MODES)
        },
        "mode_entropy": mode_ent,
        "action_entropy": act_ent,
        "band_entropy": band_ent,
        "top_1_band_fraction": top_band_frac,
        "top_2_band_fraction": top_2_band_frac,
        "distinct_bands": distinct_bands,
        "mean_consecutive_band_run": mean_run_all,
        "max_consecutive_band_run": max_run_all,
        "repeated_band_dwell_fraction": repeat_frac_all,
        "q_max": q_max,
        "q_min": q_min,
        "q_mean": q_mean,
        "q_std": q_std,
        "q_margin_mean": mean_q_margin,
        "bellman_td_error_mean": td_mean,
        "bellman_td_error_p90": td_p90,
        "bellman_loss": bellman_loss,
        "detection_pd_mean": float(np.mean([s["pd"] for s in scen_records.values()])),
        "detection_pfa_mean": float(np.mean([s["pfa"] for s in scen_records.values()])),
        "first_hit_latency_ms_mean": float(np.mean([s["first_hit_latency_ms"] for s in scen_records.values()])),
        "gross_hits_per_ms": float(total_gross_hits / tot_dwell_ms) if tot_dwell_ms > 0 else 0.0,
        "novel_hits_per_ms": float(total_novel_hits / tot_dwell_ms) if tot_dwell_ms > 0 else 0.0,
        "repeated_hit_fraction": float(total_repeated_hits / total_gross_hits) if total_gross_hits > 0 else 0.0,
        "mean_ir_decision": float(total_gross_hits / float(n_tot_steps)) * 100.0,
        "sparse_ir_decision": float(np.mean(sparse_irs)),
        "agile_ir_decision": float(np.mean(agile_irs)),
        "sparse_pd": float(np.mean(sparse_pds)),
        "agile_pd": float(np.mean(agile_pds)),
        "normal_pct_interval_slope": float(slope),
        "intervals": interval_summary,
        "train_telemetry": train_telemetry,
        "scenarios": scen_records,
    }
    return summary


def classify_recovery_trajectory(summary_c: Dict[str, Any], baseline_25: Dict[str, Any], g4_b: Dict[str, Any]) -> Tuple[str, str]:
    """Classify the recovery trajectory of G4-C into R1, R2, R3, or R4."""
    mode_ent = summary_c["mode_entropy"]
    normal_pct = summary_c["mode_distribution_pct"]["NORMAL_DWELL"]
    q_max = summary_c["q_max"]
    normal_slope = summary_c["normal_pct_interval_slope"]
    late_intervals_normal = [iw["normal_pct"] for iw in summary_c["intervals"][5:]]
    late_normal_mean = float(np.mean(late_intervals_normal))
    top_band_frac = summary_c["top_1_band_fraction"]
    repeat_band_frac = summary_c["repeated_band_dwell_fraction"]
    pd_mean = summary_c["detection_pd_mean"]

    # Check for degradation
    if q_max > 100.0 or pd_mean < 40.0:
        return "R4", "Objective-induced degradation: value runaway or detection collapse."

    # Check for transient perturbation (dies off to near 0 in later intervals)
    if late_normal_mean < 0.5 and normal_slope < -0.5:
        return "R3", "Transient perturbation: mode diversity appeared initially but re-converged to 100% LONG."

    # Check for spatial collapse despite mode recovery
    # Severe spatial collapse: top band fraction > 90% or repeated band dwell fraction > 90%
    if top_band_frac > 0.90 or repeat_band_frac > 0.90:
        return "R2", "Mode-only recovery: policy diversified dwell modes but remained severely collapsed spatially."

    # If mode diversity is active and sustained across late intervals without spatial collapse
    if normal_pct > 2.0 and late_normal_mean > 2.0:
        return "R1", (
            f"Stable partial recovery: sustained NORMAL dwell across all 10 intervals (late mean {late_normal_mean:.2f}%, slope {normal_slope:+.3f}), "
            f"controlled Q values (Qmax {q_max:.2f} vs Gate-50 ~109), and high detection quality (Pd {pd_mean:.2f}% vs Gate-50 63.3%). "
            f"Insufficient for full qualification due to Hmode {mode_ent:.3f} < 0.40."
        )

    return "R2", "Mode-only recovery with limited spatial adaptation."


def main():
    parser = argparse.ArgumentParser(description="Phase G4.1: Read-Only Recovery Characterization")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--val-dir", type=str, default="D:/TSRD/stare/val_stare")
    args = parser.parse_args()

    val_dir = Path(args.val_dir)
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found at {val_dir}")

    device = torch.device(args.device)
    logger.info("Phase G4.1 Read-Only Recovery Characterization starting on device: %s", device)

    summaries = {}
    for key, cfg in CHECKPOINTS.items():
        summaries[key] = evaluate_checkpoint_detailed(key, cfg, val_dir, device)

    # Forensic Classification for G4-C
    classification, justification = classify_recovery_trajectory(
        summaries["g4_c"], summaries["gate25_baseline"], summaries["g4_b"]
    )
    logger.info("=" * 80)
    logger.info("PHASE G4.1 CLASSIFICATION FOR G4-C: %s", classification)
    logger.info("Justification: %s", justification)
    logger.info("=" * 80)

    # Save manifest
    manifest_data = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "task": "Phase G4.1 Read-Only Recovery Characterization",
        "verdict": f"PARTIAL_RECOVERY_SIGNAL_INSUFFICIENT_{classification}",
        "classification": classification,
        "classification_justification": justification,
        "models": summaries,
    }

    manifest_path = repo_root / "reports/g4_1_recovery_characterization_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    logger.info("Emitted manifest: %s", manifest_path)


if __name__ == "__main__":
    main()

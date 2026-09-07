"""Soft-Blend MoE Hyperparameter Sweep (zero-training diagnostic).

Sweeps w_rev in {0.3,0.4,0.5,0.6,0.7,0.8} and tau in {0.05,0.10,0.15,0.20,0.25}
over checkpoint_gate_225000.pt using pure Boltzmann softmax sampling on fused scores.
No training -- frozen weights only.

Evaluation protocol (permanently adopted):
  - Primary metric: Boltzmann tau-sampled intercept rate
  - Gate criterion: >=6/10 individual scenario wins vs. RoundRobin
  - Anti-reward-hacking: high entropy + 0% intercepts = FAIL (not pass)

Outputs:
  - checkpoints/scheduler_clean_staged/soft_blend_sweep_report.json
  - Live progress to stdout as sweep runs
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.contracts import CANONICAL_N_BANDS, CANONICAL_N_MODES
from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from src.environment.scenario_generator import load_h5_records
from src.models.drqn_scheduler import DRQNScheduler
from src.models.smartscan_moe import SmartScanMoE
from src.telemetry.schema import coerce, shannon_entropy
from src.training.val_set import FixedValidationSet

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("soft_blend_sweep")

# Ground-truth RoundRobin baselines (from generalization_gate_225k_report.json)
ROUNDROBIN_BASELINE: dict[str, float] = {
    "config_117": 3.300,
    "config_119": 0.400,
    "config_143": 0.300,
    "config_194": 4.400,
    "config_195": 7.100,
    "config_241": 1.100,
    "config_29":  0.500,
    "config_42":  2.400,
    "config_64":  12.000,
    "config_96":  3.400,
}

TIE_TOLERANCE_PCT = 0.1
ENTROPY_HACK_GUARD = 3.0   # bits: below this, entropy improvement is genuine
INTERCEPT_HACK_FLOOR = 0.05  # pct: above this, intercepts are real


def boltzmann_sample(scores: np.ndarray, tau: float, rng: np.random.Generator) -> int:
    """Sample action from Boltzmann distribution over fused scores."""
    if tau <= 0.0:
        return int(np.argmax(scores))
    shifted = scores / float(tau)
    shifted -= shifted.max()
    probs = np.exp(shifted)
    probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))


def run_one_episode(
    drqn: DRQNScheduler,
    records: list[dict[str, Any]],
    env_config: dict[str, Any],
    w_rev: float,
    tau: float,
    seed: int,
    n_steps: int = 1000,
) -> dict[str, Any]:
    """Run one 1000-step episode of Soft-Blend MoE with Boltzmann sampling.

    fused = (1-w_rev)*eager_norm + w_rev*revisit_norm
    action ~ Boltzmann(fused, tau)
    """
    n_bands = int(env_config.get("n_bands", CANONICAL_N_BANDS))
    n_modes = int(env_config.get("n_modes", CANONICAL_N_MODES))
    w_eager = 1.0 - w_rev
    rng = np.random.default_rng(seed)

    env = CognitiveRFScanEnv(env_config, records=records, seed=seed)
    obs, _ = env.reset()

    moe_cfg = {
        "n_bands": n_bands,
        "n_modes": n_modes,
        "n_actions": n_bands * n_modes,
        "eager_weight": w_eager,
        "revisit_weight": w_rev,
        "preemptive_weight": 0.0,
        "semantic_weight": 0.0,
        "decay_rate": 0.05,
        "max_revisit_gap": 200,
        "device": "cpu",
    }
    moe = SmartScanMoE(drqn, moe_cfg)
    moe.reset()

    ep_reward = 0.0
    ep_hits = 0
    band_counts = np.zeros(n_bands, dtype=int)
    hidden = None

    for step in range(n_steps):
        fused, eager_norm, revisit_norm, hidden, obs_1d, raw_q = moe._compute_fused_full(
            np.asarray(obs, dtype=np.float32), eager_hidden=hidden
        )
        action = boltzmann_sample(fused, tau=tau, rng=rng)
        band = int(action // n_modes)
        band_counts[band] += 1

        obs, reward, term, trunc, info = env.step(action)
        ep_reward += float(reward)
        ep_hits += int(info.get("hit", False))
        moe.revisit_agent.update(band)

        if term or trunc:
            break

    steps_done = max(1, step + 1)
    intercept_pct = (ep_hits / steps_done) * 100.0
    distinct_bands = int(np.count_nonzero(band_counts))
    band_ent = float(shannon_entropy(band_counts))

    return {
        "intercept_pct": float(intercept_pct),
        "hits": int(ep_hits),
        "steps": int(steps_done),
        "distinct_bands": distinct_bands,
        "band_entropy": float(band_ent),
        "total_reward": float(ep_reward),
        "avg_reward": float(ep_reward / steps_done),
    }


def wlt_vs_rr(mean_pct: float, scenario_id: str) -> str:
    """Return 'W', 'L', or 'T' vs the canonical RoundRobin baseline."""
    rr = ROUNDROBIN_BASELINE.get(scenario_id, 0.0)
    if mean_pct >= rr + TIE_TOLERANCE_PCT:
        return "W"
    elif mean_pct <= rr - TIE_TOLERANCE_PCT:
        return "L"
    else:
        return "T"


def run_sweep(
    data_dir: str = "D:/TSRD",
    ckpt_path: str = "checkpoints/scheduler_clean_staged/checkpoint_gate_225000.pt",
    model_cfg_path: str = "configs/model_config.yaml",
    train_cfg_path: str = "configs/training_config.yaml",
    n_scenarios: int = 10,
    steps_per_scenario: int = 1000,
    seeds: tuple[int, ...] = (42, 123, 999, 1701, 2049),
    w_rev_values: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
    tau_values: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25),
    output_path: str = "checkpoints/scheduler_clean_staged/soft_blend_sweep_report.json",
) -> dict[str, Any]:
    """Run the full hyperparameter sweep."""

    with open(model_cfg_path, encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)
    with open(train_cfg_path, encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)

    drqn_cfg = full_cfg.get("drqn_scheduler", {})
    env_cfg = {
        **train_cfg.get("environment", {}),
        **full_cfg.get("reward", {}),
        "semantic_memory_path": ":memory:",
    }
    env_cfg.setdefault("n_bands", drqn_cfg.get("n_bands", CANONICAL_N_BANDS))
    env_cfg.setdefault("n_modes", drqn_cfg.get("n_modes", CANONICAL_N_MODES))
    env_cfg.setdefault("n_actions", drqn_cfg.get("n_actions", 180))

    print(f"\nLoading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    drqn = DRQNScheduler(
        obs_dim=int(drqn_cfg.get("obs_dim", 360)),
        n_bands=int(env_cfg["n_bands"]),
        n_actions=int(env_cfg["n_actions"]),
        n_modes=int(env_cfg["n_modes"]),
        lstm_hidden=int(drqn_cfg.get("lstm_hidden", 64)),
        lstm_layers=int(drqn_cfg.get("lstm_layers", 1)),
    )
    if "state_dict" in ckpt:
        drqn.load_state_dict(ckpt["state_dict"])
    drqn.eval()
    for p in drqn.parameters():
        p.requires_grad_(False)
    global_step = int(ckpt.get("global_step", 225000))
    print(f"Checkpoint loaded: global_step={global_step}, reward_baseline={ckpt.get('reward_baseline', 'N/A')}")

    print(f"\nLoading {n_scenarios} validation scenarios from {data_dir}/stare/val_stare")
    val_set = FixedValidationSet(
        data_root=data_dir,
        subset="val",
        mode="stare",
        n_files=n_scenarios,
        seed=42,
        freq_min_mhz=float(env_cfg.get("freq_min_mhz", 0.0)),
        freq_max_mhz=float(env_cfg.get("freq_max_mhz", 18000.0)),
        time_horizon_us=float(env_cfg.get("time_horizon_us", 0.0)) or None,
        max_pulses=int(env_cfg.get("max_pulses", 50000)),
        allow_synthetic_fallback=False,
    )
    scenario_files = val_set.files_used
    print(f"Scenarios: {[sid for _, sid, _ in scenario_files]}")

    scenarios_data: list[tuple[str, list[dict[str, Any]]]] = []
    for path, sid, _ in scenario_files:
        print(f"  Preloading {sid}...")
        recs = load_h5_records(
            path,
            freq_min_mhz=float(env_cfg.get("freq_min_mhz", 0.0)),
            freq_max_mhz=float(env_cfg.get("freq_max_mhz", 18000.0)),
            time_horizon_us=float(env_cfg.get("time_horizon_us", 0.0)) or None,
            max_pulses=int(env_cfg.get("max_pulses", 50000)),
        )
        scenarios_data.append((sid, recs))

    scenario_ids = [sid for sid, _ in scenarios_data]
    n_combos = len(w_rev_values) * len(tau_values)
    total_episodes = n_combos * len(scenario_ids) * len(seeds)
    print(f"\n{'='*80}")
    print(f"SWEEP: {n_combos} configs x {len(scenario_ids)} scenarios x {len(seeds)} seeds = {total_episodes} episodes")
    print(f"w_rev grid: {w_rev_values}")
    print(f"tau grid:   {tau_values}")
    print(f"{'='*80}\n")

    all_config_results: list[dict[str, Any]] = []
    episode_count = 0

    for w_rev, tau in product(w_rev_values, tau_values):
        config_label = f"w_rev={w_rev:.1f},tau={tau:.2f}"
        per_scenario_means: dict[str, dict[str, float]] = {}

        for sid, recs in scenarios_data:
            seed_results = []
            for seed in seeds:
                result = run_one_episode(
                    drqn=drqn,
                    records=recs,
                    env_config=env_cfg,
                    w_rev=w_rev,
                    tau=tau,
                    seed=seed,
                    n_steps=steps_per_scenario,
                )
                seed_results.append(result)
                episode_count += 1

            mean_intercept = float(np.mean([r["intercept_pct"] for r in seed_results]))
            per_scenario_means[sid] = {
                "intercept_pct": mean_intercept,
                "hits": float(np.mean([r["hits"] for r in seed_results])),
                "band_entropy": float(np.mean([r["band_entropy"] for r in seed_results])),
                "distinct_bands": float(np.mean([r["distinct_bands"] for r in seed_results])),
                "total_reward": float(np.mean([r["total_reward"] for r in seed_results])),
                "rr_baseline": ROUNDROBIN_BASELINE.get(sid, 0.0),
                "wlt": wlt_vs_rr(mean_intercept, sid),
            }

        all_intercepts = [v["intercept_pct"] for v in per_scenario_means.values()]
        mean_intercept_all = float(np.mean(all_intercepts))
        std_intercept_all = float(np.std(all_intercepts))
        mean_entropy_all = float(np.mean([v["band_entropy"] for v in per_scenario_means.values()]))

        wins = sum(1 for v in per_scenario_means.values() if v["wlt"] == "W")
        losses = sum(1 for v in per_scenario_means.values() if v["wlt"] == "L")
        ties = sum(1 for v in per_scenario_means.values() if v["wlt"] == "T")

        reward_hack_flag = (mean_intercept_all < INTERCEPT_HACK_FLOOR) and (mean_entropy_all > ENTROPY_HACK_GUARD)
        passes_gate = (wins >= 6) and (not reward_hack_flag)

        config_result = {
            "w_rev": float(w_rev),
            "w_eager": float(1.0 - w_rev),
            "tau": float(tau),
            "label": config_label,
            "mean_intercept_pct": mean_intercept_all,
            "std_intercept_pct": std_intercept_all,
            "mean_band_entropy": mean_entropy_all,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "wlt_string": f"{wins}W-{losses}L-{ties}T",
            "passes_gate": bool(passes_gate),
            "reward_hack_flagged": bool(reward_hack_flag),
            "per_scenario": per_scenario_means,
        }
        all_config_results.append(config_result)

        gate_str = "PASS" if passes_gate else ("HACK" if reward_hack_flag else "fail")
        print(
            f"  [{config_label}]  {mean_intercept_all:5.2f}+/-{std_intercept_all:4.2f}%  "
            f"{wins}W-{losses}L-{ties}T vs RR  entropy={mean_entropy_all:.2f}b  {gate_str}"
        )

    ranked = sorted(all_config_results, key=lambda r: (-r["wins"], -r["mean_intercept_pct"]))
    best = ranked[0]

    print(f"\n{'='*100}")
    print(f"SOFT-BLEND SWEEP RESULTS  checkpoint gate_{global_step}")
    print(f"{'='*100}")
    print(f"{'Config':<24} | {'Mean%':>7} | {'Std%':>6} | {'W-L-T':>9} | {'Entropy':>8} | Gate")
    print("-" * 80)
    for r in ranked:
        gate_str = "PASS" if r["passes_gate"] else ("HACK" if r["reward_hack_flagged"] else "fail")
        print(
            f"  {r['label']:<22} | {r['mean_intercept_pct']:6.2f}% | {r['std_intercept_pct']:5.2f}% | "
            f"  {r['wlt_string']:<7} | {r['mean_band_entropy']:7.2f}b | {gate_str}"
        )
    print(f"{'='*100}")
    print(f"\nBEST: {best['label']}  {best['mean_intercept_pct']:.2f}%+/-{best['std_intercept_pct']:.2f}%  {best['wlt_string']}  gate={'PASS' if best['passes_gate'] else 'FAIL'}")

    print(f"\nPer-scenario breakdown (best = {best['label']}):")
    for sid in scenario_ids:
        scen = best["per_scenario"].get(sid, {})
        rr = ROUNDROBIN_BASELINE.get(sid, 0.0)
        pct = scen.get("intercept_pct", 0.0)
        wlt = scen.get("wlt", "?")
        print(f"  {sid:<14}  sweep={pct:6.2f}%  RR={rr:5.2f}%  {wlt}")

    passing = [r for r in ranked if r["passes_gate"]]
    print(f"\nConfigs passing gate (>=6/10 wins): {len(passing)}/{len(all_config_results)}")
    if passing:
        print("GATE CLEARED -- recommend Stage D with best passing config.")
    else:
        print("GATE NOT CLEARED -- recommend scoping Stage D learned meta-controller.")

    report = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "checkpoint": str(ckpt_path),
        "global_step": global_step,
        "n_scenarios": len(scenarios_data),
        "scenarios": scenario_ids,
        "steps_per_scenario": steps_per_scenario,
        "seeds": list(seeds),
        "w_rev_grid": list(w_rev_values),
        "tau_grid": list(tau_values),
        "n_episodes_total": episode_count,
        "roundrobin_baselines": ROUNDROBIN_BASELINE,
        "gate_criterion": ">=6/10 individual scenario wins vs RoundRobin",
        "anti_reward_hacking_guard": {
            "entropy_threshold": ENTROPY_HACK_GUARD,
            "intercept_floor": INTERCEPT_HACK_FLOOR,
            "description": "flag if entropy>3.0b AND intercept<0.05% (high dispersion, zero hits = phase-lock broken but no real gain)",
        },
        "ranked_results": ranked,
        "best_config": best["label"],
        "best_wins": best["wins"],
        "best_mean_intercept_pct": best["mean_intercept_pct"],
        "best_passes_gate": best["passes_gate"],
        "n_configs_passing_gate": len(passing),
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coerce(report), f, indent=2)
    print(f"\nReport saved: {out_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Soft-Blend MoE hyperparameter sweep (zero-training).")
    parser.add_argument("--data-dir", type=str, default="D:/TSRD")
    parser.add_argument("--ckpt", type=str, default="checkpoints/scheduler_clean_staged/checkpoint_gate_225000.pt")
    parser.add_argument("--n-scenarios", type=int, default=10)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--out", type=str, default="checkpoints/scheduler_clean_staged/soft_blend_sweep_report.json")
    parser.add_argument("--w-rev", type=str, default="0.3,0.4,0.5,0.6,0.7,0.8", help="Comma-separated w_rev values")
    parser.add_argument("--tau", type=str, default="0.05,0.10,0.15,0.20,0.25", help="Comma-separated tau values")
    parser.add_argument("--seeds", type=str, default="42,123,999,1701,2049", help="Comma-separated seeds")
    args = parser.parse_args()

    run_sweep(
        data_dir=args.data_dir,
        ckpt_path=args.ckpt,
        n_scenarios=args.n_scenarios,
        steps_per_scenario=args.steps,
        seeds=tuple(int(x) for x in args.seeds.split(",")),
        w_rev_values=tuple(float(x) for x in args.w_rev.split(",")),
        tau_values=tuple(float(x) for x in args.tau.split(",")),
        output_path=args.out,
    )

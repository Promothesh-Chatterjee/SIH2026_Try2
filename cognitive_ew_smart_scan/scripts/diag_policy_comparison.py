"""RC-2 controlled diagnostic: Random vs RoundRobin vs DRQN/MoE.

Runs each policy over the SAME fixed validation scenarios (FixedValidationSet)
and emits a metrics table (Pd, Pfa, coverage, discovery, intercept rate,
avg intercept time, avg reward/step, action/mode entropy) plus the reward
decomposition and — for the DRQN/MoE leg — MoE attribution aggregates.

Usage:
  python scripts/diag_policy_comparison.py --data-dir D:/TSRD \\
      --ckpt cognitive_ew_smart_scan/checkpoints/scheduler/best.pt \\
      --n-files 1 --out runs/diag_1k.json
No model weights are modified; this is read-only over the environment.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.contracts import DWELL_MODES, NORMAL_DWELL  # noqa: E402
from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv  # noqa: E402
from src.models.deinterleaver import PDWTransformerEncoder  # noqa: E402
from src.models.drqn_scheduler import DRQNScheduler  # noqa: E402
from src.models.smartscan_moe import SmartScanMoE  # noqa: E402
from src.preprocessing.normalise import load_normalization_stats  # noqa: E402
from src.telemetry.schema import coerce, shannon_entropy  # noqa: E402
from src.training.val_set import FixedValidationSet  # noqa: E402

logger = logging.getLogger("diag")


def load_env_builders(data_dir, model_cfg_path="configs/model_config.yaml", cfg_path="configs/smoke_rc2_config.yaml"):
    full = yaml.safe_load(open(model_cfg_path, encoding="utf-8"))
    tr = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    drqn_cfg = full["drqn_scheduler"]
    env_cfg = {**tr["environment"], **full["reward"]}
    env_cfg.setdefault("n_bands", drqn_cfg.get("n_bands", 36))
    env_cfg.setdefault("n_modes", drqn_cfg.get("n_modes", 5))
    env_cfg.setdefault("n_actions", drqn_cfg.get("n_actions", 180))
    deinter = None
    fit_stats = None
    dpath = tr.get("deinterleaver_ckpt", "checkpoints/deinterleaver/best.pt")
    if Path(dpath).exists():
        d_cfg = full.get("deinterleaver", {})
        deinter = PDWTransformerEncoder(
            pdw_dim=d_cfg.get("pdw_dim", 6), d_model=d_cfg.get("d_model", 128),
            nhead=d_cfg.get("nhead", 8), num_layers=d_cfg.get("num_layers", 4),
            dim_feedforward=d_cfg.get("dim_feedforward", 512), dropout=d_cfg.get("dropout", 0.1),
            embed_dim=d_cfg.get("embed_dim", 64),
        )
        state = torch.load(dpath, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        deinter.load_state_dict(state, strict=False)
        deinter.eval()
        nstat = tr.get("normalization_stats", "checkpoints/deinterleaver/normalization_stats.json")
        if Path(nstat).exists():
            fit_stats = load_normalization_stats(nstat)
    return full, tr, env_cfg, deinter, fit_stats


def make_env(data_dir, env_cfg, provider, deinter, fit_stats, seed=42):
    return CognitiveRFScanEnv(
        env_cfg, records=None, seed=seed, records_provider=provider,
        deinterleaver_model=deinter, deinterleaver_config={"fit_stats": fit_stats} if fit_stats else {},
    )


def run_episode(env, policy, drqn=None, moe=None, hidden=None, max_steps=1000, collect_attr=False):
    band_counts = np.zeros(env.n_bands, dtype=np.float64)
    mode_counts = np.zeros(env.n_modes, dtype=np.float64)
    action_counts = np.zeros(env.action_space.n, dtype=np.float64)
    attr_acc = {"n": 0, "same_argmax": 0, "drqn_rank": 0.0, "q": 0.0, "eager": 0.0,
                "revisit": 0.0, "semantic": 0.0, "preempt": 0.0, "fused": 0.0}
    q_argmax_band_counts = np.zeros(env.n_bands, dtype=np.float64)
    obs, _ = env.reset()
    done = False
    steps = 0
    if moe is not None:
        moe.reset()
        if hidden is not None:
            moe.eager_agent.hidden = hidden
    while not done and steps < max_steps:
        if policy == "random":
            action = int(np.random.randint(0, env.action_space.n))
            attr = None
        elif policy == "roundrobin":
            band = steps % env.n_bands
            action = int(band * env.n_modes + NORMAL_DWELL)
            attr = None
        else:  # drqn_moe
            with torch.inference_mode():
                action, hidden, attr = moe.select_action(obs, hidden)
        obs, rew, term, trunc, info = env.step(action)
        band_counts[int(action // env.n_modes)] += 1
        mode_counts[int(action % env.n_modes)] += 1
        action_counts[int(action)] += 1
        if attr is not None and collect_attr:
            attr_acc["n"] += 1
            attr_acc["same_argmax"] += int(bool(attr.get("same_argmax", False)))
            attr_acc["drqn_rank"] += float(attr.get("drqn_rank", 0.0) or 0.0)
            attr_acc["q"] += float(attr.get("q_score") or 0.0)
            attr_acc["eager"] += float(attr.get("eager_score") or 0.0)
            attr_acc["revisit"] += float(attr.get("revisit_score") or 0.0)
            attr_acc["semantic"] += float(attr.get("semantic_score") or 0.0)
            attr_acc["preempt"] += float(attr.get("preemptive_score") or 0.0)
            attr_acc["fused"] += float(attr.get("fused_score") or 0.0)
            qab = attr.get("q_argmax_band")
            if qab is not None and 0 <= int(qab) < env.n_bands:
                q_argmax_band_counts[int(qab)] += 1
        done = bool(term or trunc)
        steps += 1
    fom = env.get_fom()
    if attr_acc["n"]:
        n = attr_acc["n"]
        attr_acc = {
            "n": n,
            "same_argmax_fraction": attr_acc["same_argmax"] / n,
            "mean_drqn_rank": attr_acc["drqn_rank"] / n,
            "mean_q_score": attr_acc["q"] / n,
            "mean_eager_score": attr_acc["eager"] / n,
            "mean_revisit_score": attr_acc["revisit"] / n,
            "mean_semantic_score": attr_acc["semantic"] / n,
            "mean_preemptive_score": attr_acc["preempt"] / n,
            "mean_fused_score": attr_acc["fused"] / n,
            "q_argmax_band": int(np.argmax(q_argmax_band_counts)) if attr_acc["n"] else None,
            "selected_band": int(np.argmax(band_counts)) if steps else None,
        }
    else:
        attr_acc = None
    return {
        "fom": fom,
        "steps": steps,
        "band_counts": band_counts,
        "mode_counts": mode_counts,
        "action_counts": action_counts,
        "attr": attr_acc,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="D:/TSRD")
    ap.add_argument("--ckpt", default="checkpoints/scheduler/best.pt")
    ap.add_argument("--n-files", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--policies", nargs="+", default=["random", "roundrobin", "drqn_moe"])
    args = ap.parse_args()

    full, tr, env_cfg, deinter, fit_stats = load_env_builders(args.data_dir)
    vs = FixedValidationSet(
        data_root=args.data_dir, subset="val", mode="stare", n_files=args.n_files, seed=args.seed,
        freq_min_mhz=float(env_cfg.get("freq_min_mhz", 0.0)),
        freq_max_mhz=float(env_cfg.get("freq_max_mhz", 18000.0)),
        time_horizon_us=float(env_cfg.get("time_horizon_us", 0.0)) or None,
        max_pulses=int(env_cfg.get("max_pulses", 50000)),
        allow_synthetic_fallback=False,
    )
    logger.info("Fixed scenarios: %s", [sid for _, sid, _ in vs.files_used])

    drqn = None
    moe = None
    if "drqn_moe" in args.policies:
        drqn_cfg = full["drqn_scheduler"]
        drqn = DRQNScheduler(
            obs_dim=drqn_cfg["obs_dim"], n_bands=drqn_cfg["n_bands"], n_actions=drqn_cfg["n_actions"],
            lstm_hidden=drqn_cfg["lstm_hidden"], lstm_layers=drqn_cfg["lstm_layers"],
        )
        state = torch.load(args.ckpt, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = drqn.load_state_dict(state, strict=False)
        if missing:
            logger.warning("DRQN checkpoint missing keys: %d", len(missing))
        drqn.eval()
        moe = SmartScanMoE(drqn, {**full["smartscan_moe"], "n_bands": drqn_cfg["n_bands"],
                                  "n_modes": drqn_cfg["n_modes"], "n_actions": drqn_cfg["n_actions"]})

    results = {}
    for policy in args.policies:
        hidden = None
        if policy == "drqn_moe":
            hidden = drqn.init_hidden(1, "cpu")
        env = make_env(args.data_dir, env_cfg, vs.sample, deinter, fit_stats, seed=args.seed)
        ep_results = []
        for _ in range(len(vs.files_used)):
            if policy == "drqn_moe":
                hidden = drqn.init_hidden(1, "cpu") if hidden is None else hidden
            r = run_episode(env, policy, drqn=drqn, moe=moe, hidden=hidden,
                            collect_attr=(policy == "drqn_moe"))
            ep_results.append(r)

        agg = {
            "policy": policy,
            "n_episodes": len(ep_results),
            "steps": sum(r["steps"] for r in ep_results),
            "hits": sum(int(r["fom"]["n_hits"]) for r in ep_results),
            "Pd": float(np.mean([r["fom"]["Pd"] for r in ep_results])),
            "Pfa": float(np.mean([r["fom"]["Pfa"] for r in ep_results])),
            "coverage": float(np.mean([r["fom"]["band_selection_coverage"] for r in ep_results])),
            "discovery_rate": float(np.mean([r["fom"]["discovery_rate"] for r in ep_results])),
            "intercept_rate": float(np.mean([r["fom"]["avg_intercept_rate"] for r in ep_results])),
            "avg_intercept_time_us": float(np.mean(
                [r["fom"]["avg_intercept_time_error_us"] for r in ep_results if r["fom"]["n_hits"] > 0])),
            "avg_reward_per_step": float(np.mean([r["fom"]["avg_reward"] for r in ep_results])),
            "action_entropy": float(np.mean([shannon_entropy(r["action_counts"]) for r in ep_results])),
            "band_entropy": float(np.mean([shannon_entropy(r["band_counts"]) for r in ep_results])),
            "mode_entropy": float(np.mean([shannon_entropy(r["mode_counts"]) for r in ep_results])),
        }
        for k in ("reward_novel", "reward_hit", "reward_miss", "reward_dwell_cost", "reward_false_alarm",
                  "reward_timing", "reward_priority", "reward_info_gain", "reward_redundant", "reward_delay"):
            key = {"reward_novel": "avg_reward_novel_term", "reward_hit": "avg_reward_hit_term",
                   "reward_miss": "avg_reward_miss_penalty", "reward_dwell_cost": "avg_reward_dwell_cost",
                   "reward_false_alarm": "avg_reward_false_alarm_penalty", "reward_timing": "avg_reward_timing_penalty",
                   "reward_priority": "avg_reward_priority_term", "reward_info_gain": "avg_reward_info_gain_term",
                   "reward_redundant": "avg_reward_redundant_penalty", "reward_delay": "avg_reward_delay_penalty"}[k]
            vals = [r["fom"][key] for r in ep_results if key in r["fom"]]
            agg["per_step_" + k] = float(np.mean(vals)) if vals else None

        if policy == "drqn_moe":
            attrs = [r["attr"] for r in ep_results if r["attr"] is not None]
            if attrs:
                for k in ("same_argmax_fraction", "mean_drqn_rank", "mean_q_score", "mean_eager_score",
                          "mean_revisit_score", "mean_semantic_score", "mean_preemptive_score",
                          "mean_fused_score"):
                    agg["moe_" + k] = float(np.mean([a[k] for a in attrs]))
                agg["moe_q_argmax_band_union"] = sorted(set(a["q_argmax_band"] for a in attrs if a["q_argmax_band"] is not None))
                agg["moe_selected_band_union"] = sorted(set(a["selected_band"] for a in attrs if a["selected_band"] is not None))
            agg["moe_weights"] = {k: float(getattr(moe, k)) for k in
                                  ("eager_weight", "revisit_weight", "semantic_weight", "preemptive_weight")}

        results[policy] = agg

    # Print compact table.
    header = ["policy", "steps", "hits", "Pd", "Pfa", "cov", "disc", "int_rate", "int_time_us",
              "rew/step", "A_ent", "M_ent"]
    print("\n" + " | ".join(header))
    for p, a in results.items():
        row = [str(x) for x in [p, a["steps"], a["hits"], f"{a['Pd']:.3f}", f"{a['Pfa']:.3f}", f"{a['coverage']:.3f}",
               f"{a['discovery_rate']:.3f}", f"{a['intercept_rate']:.3f}", f"{a['avg_intercept_time_us']:.1f}",
               f"{a['avg_reward_per_step']:.4f}", f"{a['action_entropy']:.2f}", f"{a['mode_entropy']:.2f}"]]
        print(" | ".join(row))
    print("\nPer-step reward decomposition (avg/step):")
    for p, a in results.items():
        parts = " ".join(f"{k.split('_',1)[1]}={a[k]:+.4f}" for k in sorted(a) if k.startswith("per_step_"))
        print(f"  {p}: {parts}")
    if "drqn_moe" in results:
        a = results["drqn_moe"]
        print("\nDRQN/MoE attribution:")
        print(f"  weights eager={a['moe_weights']['eager_weight']} revisit={a['moe_weights']['revisit_weight']} "
              f"semantic={a['moe_weights']['semantic_weight']} preempt={a['moe_weights']['preemptive_weight']}")
        print(f"  same_argmax_fraction={a.get('moe_same_argmax_fraction'):.4f} "
              f"mean_drqn_rank={a.get('moe_mean_drqn_rank'):.2f}")
        print(f"  semantic/eager/revisit/preempt shares = "
              f"{a.get('moe_mean_semantic_score'):.3f}/{a.get('moe_mean_eager_score'):.3f}/"
              f"{a.get('moe_mean_revisit_score'):.3f}/{a.get('moe_mean_preemptive_score'):.3f} "
              f"(fused={a.get('moe_mean_fused_score'):.3f})")
        print(f"  q_argmax bands {a.get('moe_q_argmax_band_union')} vs selected bands {a.get('moe_selected_band_union')}")

    out = args.out
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scenarios": [sid for _, sid, _ in vs.files_used],
            "validation_set_id": vs.validation_set_id,
            "checkpoint": args.ckpt,
            "results": coerce(results),
        }
        Path(out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
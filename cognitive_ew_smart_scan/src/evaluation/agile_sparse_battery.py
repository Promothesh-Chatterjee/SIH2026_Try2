"""
Phase 5 Agile and Sparse Emitter Evaluation Battery.

Runs a dedicated, reproducible benchmark across 6 canonical validation scenarios:
- config_119: High-agility frequency-hopping radar
- config_143: Sparse radar environment (low duty-cycle)
- config_241: High-agility multi-emitter (14 bands)
- config_29:  Agile hopping (12 bands)
- config_195: Dense agile multi-emitter (16 bands)
- config_64:  Dense multi-emitter stationary benchmark

Evaluates 5 policies under identical conditions with zero ground-truth leakage:
1. drqn: Pure standalone greedy DRQN (flat_argmax, zero MoE, zero heuristics)
2. random: Uniform random scheduler
3. round_robin: Deterministic round-robin across 36 bands
4. highest_occupancy: Greedy highest estimated occupancy heuristic
5. highest_uncertainty: Information-seeking highest uncertainty heuristic
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..contracts import CANONICAL_N_BANDS, CANONICAL_N_MODES, DWELL_MODES
from ..environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from ..environment.scenario_generator import load_h5_records
from ..models.baseline_suite import build_baseline
from ..models.drqn_scheduler import DRQNScheduler
from ..telemetry.schema import coerce, shannon_entropy
from ..training.policy_collapse_detector import PolicyCollapseDetector
from .canonical_metrics import EvaluationManifest, compute_canonical_metrics

logger = logging.getLogger(__name__)

AGILE_SPARSE_SCENARIOS = (
    "config_119",
    "config_143",
    "config_241",
    "config_29",
    "config_195",
    "config_64",
)

TARGET_POLICIES = (
    "drqn",
    "random",
    "round_robin",
    "highest_occupancy",
    "highest_uncertainty",
)


def run_agile_sparse_battery(
    checkpoint_path: str | Path | None = None,
    drqn_model: DRQNScheduler | None = None,
    data_dir: str | Path = "D:/TSRD",
    n_steps: int = 1000,
    seed: int = 42,
    device: str = "cpu",
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Run the 6-scenario x 5-policy agile/sparse evaluation battery.

    Args:
        checkpoint_path: Path to model checkpoint .pt (if drqn_model is None).
        drqn_model: Pre-loaded DRQNScheduler instance.
        data_dir: Root dataset directory containing stare/val_stare.
        n_steps: Evaluation steps per scenario (default: 1000).
        seed: Random seed for environment and baseline RNGs.
        device: Device for model evaluation ('cpu' recommended for determinism).
        output_json: Optional path to save evaluation results as JSON.

    Returns:
        Dictionary of results structured by scenario and policy with summary aggregates.
    """
    data_dir = Path(data_dir)
    val_stare_dir = data_dir / "stare" / "val_stare"
    if not val_stare_dir.exists():
        val_stare_dir = data_dir / "val"

    eval_device = torch.device(device)

    # Resolve DRQN model
    if drqn_model is None and checkpoint_path is not None:
        ckpt_p = Path(checkpoint_path)
        if not ckpt_p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_p}")
        ckpt = torch.load(ckpt_p, map_location=eval_device, weights_only=False)
        drqn_model = DRQNScheduler(
            obs_dim=360,
            n_bands=CANONICAL_N_BANDS,
            n_actions=CANONICAL_N_BANDS * CANONICAL_N_MODES,
            lstm_hidden=256,
            lstm_layers=2,
            n_modes=CANONICAL_N_MODES,
        ).to(eval_device)
        if "state_dict" in ckpt:
            drqn_model.load_state_dict(ckpt["state_dict"])
        drqn_model.eval()

    # Preload records for the 6 canonical scenarios
    scenario_records: dict[str, list[Any]] = {}
    for scen_id in AGILE_SPARSE_SCENARIOS:
        h5_path = val_stare_dir / f"{scen_id}.h5"
        if not h5_path.exists():
            logger.warning("Scenario file %s not found in %s", scen_id, val_stare_dir)
            continue
        recs = load_h5_records(
            h5_path,
            freq_min_mhz=0.0,
            freq_max_mhz=18000.0,
            time_horizon_us=30000000.0,
            max_pulses=50000,
        )
        scenario_records[scen_id] = recs
        logger.info("Loaded scenario %s: %d pulses", scen_id, len(recs))

    env_base_cfg = {
        "n_bands": CANONICAL_N_BANDS,
        "n_modes": CANONICAL_N_MODES,
        "freq_min_mhz": 0.0,
        "freq_max_mhz": 18000.0,
        "ibw_mhz": 500.0,
        "dwell_time_us": 500.0,
        "detection_threshold_db": -140.0,
        "time_horizon_us": 30000000.0,
        "max_pulses": 50000,
        "max_steps_per_episode": n_steps,
        "semantic_memory_enabled": False,
        "reward": {"version": "v2"},
    }

    results: dict[str, Any] = {
        "metadata": {
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "scenarios": list(scenario_records.keys()),
            "policies": list(TARGET_POLICIES),
            "n_steps": n_steps,
            "seed": seed,
            "device": str(device),
        },
        "per_scenario": {},
        "summary": {},
    }

    for scen_id, recs in scenario_records.items():
        results["per_scenario"][scen_id] = {}
        for pol_name in TARGET_POLICIES:
            # Seed and build environment
            env_cfg = copy.deepcopy(env_base_cfg)
            env = CognitiveRFScanEnv(
                env_cfg,
                records=recs,
                seed=seed,
                semantic_memory_path=":memory:",
            )
            obs, _ = env.reset()

            # Build policy
            agent = build_baseline(
                pol_name,
                n_bands=CANONICAL_N_BANDS,
                n_modes=CANONICAL_N_MODES,
                drqn=drqn_model,
                config={"action_selection_mode": "flat_argmax", "tau": 0.0},
                seed=seed,
                device=str(device),
            )
            if hasattr(agent, "reset"):
                agent.reset()

            hidden = None
            if hasattr(agent, "init_hidden"):
                hidden = agent.init_hidden(1, device)

            ep_hits = 0
            first_detection_time_us: float | None = None
            band_counts = np.zeros(CANONICAL_N_BANDS, dtype=np.int64)
            mode_counts = np.zeros(CANONICAL_N_MODES, dtype=np.int64)
            action_counts = np.zeros(CANONICAL_N_BANDS * CANONICAL_N_MODES, dtype=np.int64)

            for step in range(n_steps):
                if hasattr(agent, "select_action"):
                    action, hidden, attr = agent.select_action(obs, hidden)
                elif hasattr(agent, "act"):
                    action, attr = agent.act(obs)
                else:
                    action = agent.step(obs)

                action = int(action)
                band = int(action // CANONICAL_N_MODES)
                mode = int(action % CANONICAL_N_MODES)
                band_counts[band] += 1
                mode_counts[mode] += 1
                action_counts[action] += 1

                obs, reward, term, trunc, info = env.step(action)
                hit = bool(info.get("hit", False))
                if hit:
                    ep_hits += 1
                    if first_detection_time_us is None:
                        first_detection_time_us = float(env.receiver.current_time_us)

                if hasattr(agent, "update_result"):
                    agent.update_result(hit, band)
                if hasattr(agent, "update"):
                    agent.update(action)

                if term or trunc:
                    break

            steps_done = max(1, step + 1)
            fom = env.get_fom()

            manifest = EvaluationManifest(
                scenario_id=scen_id,
                seed=seed,
                episode_steps=steps_done,
                checkpoint_file=str(checkpoint_path) if checkpoint_path else None,
                total_pulses=len(recs),
                active_emitters=len(getattr(env.fom, "all_active_emitters", set())),
                discovered_emitters=len(getattr(env.fom, "discovered_emitters", set())),
                total_interceptions=ep_hits,
                total_opportunities=int(fom.get("tp", 0) + fom.get("fn", 0)),
                false_alarms=int(fom.get("fp", 0)),
                latency_samples_count=len(env.fom.intercept_time_errors),
            )

            canon = compute_canonical_metrics(
                steps_done=steps_done,
                ep_hits=ep_hits,
                tp=int(fom.get("tp", 0)),
                fn=int(fom.get("fn", 0)),
                fp=int(fom.get("fp", 0)),
                tn=int(fom.get("tn", 0)),
                band_counts=band_counts,
                action_counts=action_counts,
                mode_counts=mode_counts,
                time_errors_us=env.fom.intercept_time_errors,
                first_detection_time_us=first_detection_time_us,
                discovered_emitters=list(getattr(env.fom, "discovered_emitters", set())),
                all_active_emitters=list(getattr(env.fom, "all_active_emitters", set())),
                selected_active_opportunities=int(fom.get("selected_active_opportunities", 0)),
                spectrum_active_opportunities=int(fom.get("spectrum_active_opportunities", 0)),
                total_reward=float(sum(env.fom.rewards)) if env.fom.rewards else 0.0,
                manifest=manifest,
            )

            pol_metrics = {
                "scenario_id": scen_id,
                "policy": pol_name,
                "interception_rate": canon.interception_rate,
                "hits": canon.hits,
                "steps": canon.steps,
                "Pd": canon.pd,
                "Pfa": canon.pfa,
                "avg_intercept_time_error_us": canon.avg_intercept_time_error_us,
                "unique_bands": canon.distinct_bands,
                "unique_actions": canon.distinct_actions,
                "top_band_fraction": canon.top_band_fraction,
                "top_action_fraction": canon.top_action_fraction,
                "action_entropy": canon.action_entropy,
                "mode_entropy": canon.mode_entropy,
                "band_entropy": canon.band_entropy,
                "first_detection_time_us": canon.first_detection_time_us,
                "missed_active_emitter_count": canon.missed_active_emitter_count,
                "coverage": canon.operational_coverage,
                "discovery_rate": canon.discovery_rate,
                "manifest": manifest.to_dict(),
            }
            results["per_scenario"][scen_id][pol_name] = pol_metrics
            logger.info(
                "[%s | %s] IR=%.4f, Hits=%d, Pd=%.3f, Bands=%d, MissedEmitters=%d",
                scen_id, pol_name, canon.interception_rate, canon.hits, canon.pd, canon.distinct_bands, canon.missed_active_emitter_count
            )

    # Compute summary aggregates across the 6 scenarios
    for pol_name in TARGET_POLICIES:
        pol_scens = [
            results["per_scenario"][sid][pol_name]
            for sid in results["per_scenario"]
            if pol_name in results["per_scenario"][sid]
        ]
        if not pol_scens:
            continue

        agile_scens = [s for s in pol_scens if s["scenario_id"] in ("config_119", "config_241", "config_29", "config_195")]
        sparse_scens = [s for s in pol_scens if s["scenario_id"] in ("config_143", "config_119")]

        results["summary"][pol_name] = {
            "overall_interception_rate": float(np.mean([s["interception_rate"] for s in pol_scens])),
            "overall_hits": float(np.mean([s["hits"] for s in pol_scens])),
            "overall_pd": float(np.mean([s["Pd"] for s in pol_scens])),
            "overall_pfa": float(np.mean([s["Pfa"] for s in pol_scens])),
            "avg_latency_us": float(np.mean([s["avg_intercept_time_error_us"] for s in pol_scens])),
            "avg_unique_bands": float(np.mean([s["unique_bands"] for s in pol_scens])),
            "avg_unique_actions": float(np.mean([s["unique_actions"] for s in pol_scens])),
            "avg_action_entropy": float(np.mean([s["action_entropy"] for s in pol_scens])),
            "avg_missed_emitters": float(np.mean([s["missed_active_emitter_count"] for s in pol_scens])),
            "agile_interception_rate": float(np.mean([s["interception_rate"] for s in agile_scens])) if agile_scens else 0.0,
            "sparse_interception_rate": float(np.mean([s["interception_rate"] for s in sparse_scens])) if sparse_scens else 0.0,
        }

    # Phase 7 Policy-Collapse Diagnostic evaluation for DRQN policy
    if "drqn" in results["summary"]:
        drqn_sum = results["summary"]["drqn"]
        drqn_scens = {
            sid: results["per_scenario"][sid]["drqn"]["interception_rate"]
            for sid in results["per_scenario"]
            if "drqn" in results["per_scenario"][sid]
        }
        detector = PolicyCollapseDetector()
        collapse_diag = detector.evaluate_eval_run(
            step=0,
            distinct_bands=drqn_sum["avg_unique_bands"],
            top_band_fraction=0.0,
            top_action_fraction=0.0,
            action_entropy=drqn_sum["avg_action_entropy"],
            scenario_irs=drqn_scens,
            agile_ir=drqn_sum["agile_interception_rate"],
            sparse_ir=drqn_sum["sparse_interception_rate"],
            pd=drqn_sum["overall_pd"],
            pfa=drqn_sum["overall_pfa"],
            latency_us=drqn_sum["avg_latency_us"],
        )
        results["collapse_diagnostics"] = collapse_diag.to_dict()
        results["checkpoint_tag"] = collapse_diag.tag
        logger.info(
            "Policy-Collapse Diagnostics: severity=%s, tag=%s, reasons=%s",
            collapse_diag.severity.value, collapse_diag.tag, collapse_diag.reasons
        )
        # Phase 9B Composite Generalization Score for 6-scenario dedicated battery
        ir_overall_bat = float(drqn_sum["overall_interception_rate"])
        ir_agile_bat = float(drqn_sum["agile_interception_rate"])
        ir_sparse_bat = float(drqn_sum["sparse_interception_rate"])
        ir_worst_bat = float(min(drqn_scens.values())) if drqn_scens else 0.0
        # Average discovery rate across the 6 battery scenarios
        drqn_drs = [results["per_scenario"][sid]["drqn"].get("discovery_rate", 0.0) for sid in drqn_scens]
        dr_bat = float(np.mean(drqn_drs)) if drqn_drs else 0.0
        lat_us_bat = float(drqn_sum["avg_latency_us"] or 0.0)

        composite_score_bat = (
            ir_overall_bat
            + 0.5 * ir_agile_bat
            + 0.5 * ir_sparse_bat
            + 0.25 * ir_worst_bat
            + 0.10 * dr_bat
            - 0.0005 * lat_us_bat
        )
        drqn_sum["composite_generalization_score_battery"] = float(composite_score_bat)
        results["composite_generalization_score_battery"] = float(composite_score_bat)
        logger.info("Agile/Sparse Battery Composite Generalization Score: %.4f", composite_score_bat)

    if output_json:
        out_p = Path(output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(coerce(results), f, indent=2)
        logger.info("Saved agile/sparse battery report to %s", out_p)

    return results


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Run Agile/Sparse Evaluation Battery")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/scheduler_v2_gate5k/checkpoint_gate_5000.pt")
    parser.add_argument("--data-dir", type=str, default="D:/TSRD")
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--output", type=str, default="reports/scheduler_v2/agile_sparse_battery_5k.json")
    args = parser.parse_args()

    run_agile_sparse_battery(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        n_steps=args.n_steps,
        output_json=args.output,
    )

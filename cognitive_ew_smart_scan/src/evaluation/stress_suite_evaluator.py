"""
Deep Stress & Diagnostic Evaluation Suite for Cognitive EW Smart Scan.

Evaluates the frozen Gate 75k policy against canonical baselines across 5 stress axes:
1. Sparse Emitters (1-2 emitters)
2. Dense Emitters (15-20+ emitters)
3. Intermittent / Rotational Scan (< 1.0 pulses/ms)
4. Frequency-Agile / Hopping (multi-band hopping emitters)
5. General Unseen Test Split & Multi-Seed Generalization

Explicitly tracks:
- Per-scenario intercept rate
- Per-scenario fallback rate (DRQN Confident vs Boltzmann vs Heuristic Fallback)
- Per-scenario distinct bands and empty escape
- Per-scenario dwell mode distribution (SHORT/NORMAL/LONG/REVISIT)
- Head-to-head comparison between HighestOccupancy and DRQN+MoE
"""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..contracts import CANONICAL_N_BANDS, CANONICAL_N_MODES, DWELL_MODES
from ..environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from ..environment.scenario_generator import load_h5_records
from ..models.baseline_suite import build_baseline
from ..models.drqn_scheduler import DRQNScheduler
from ..models.smartscan_moe import SmartScanMoE
from ..telemetry.schema import shannon_entropy

logger = logging.getLogger(__name__)


# Curated representative test scenarios from D:/TSRD/ test splits
CURATED_STRESS_SCENARIOS: dict[str, list[dict[str, Any]]] = {
    "sparse": [
        {"id": "stare_config_0", "path": "D:/TSRD/stare/test_stare/config_0.h5", "desc": "1 emitter, 1.2 p/ms"},
        {"id": "stare_config_106", "path": "D:/TSRD/stare/test_stare/config_106.h5", "desc": "1 emitter, 32.3 p/ms (high pulse rate)"},
        {"id": "stare_config_131", "path": "D:/TSRD/stare/test_stare/config_131.h5", "desc": "2 emitters, 1.2 p/ms"},
        {"id": "scan_config_0", "path": "D:/TSRD/scan/test_scan/config_0.h5", "desc": "1 emitter, 0.05 p/ms (very sparse scan)"},
    ],
    "dense": [
        {"id": "stare_config_101", "path": "D:/TSRD/stare/test_stare/config_101.h5", "desc": "15 emitters, 7.6 p/ms"},
        {"id": "stare_config_105", "path": "D:/TSRD/stare/test_stare/config_105.h5", "desc": "20 emitters, 6.3 p/ms"},
        {"id": "stare_config_108", "path": "D:/TSRD/stare/test_stare/config_108.h5", "desc": "18 emitters, 4.7 p/ms"},
        {"id": "stare_config_111", "path": "D:/TSRD/stare/test_stare/config_111.h5", "desc": "17 emitters, 9.5 p/ms"},
    ],
    "intermittent_scan": [
        {"id": "scan_config_0", "path": "D:/TSRD/scan/test_scan/config_0.h5", "desc": "1 emitter, 0.05 p/ms"},
        {"id": "scan_config_115", "path": "D:/TSRD/scan/test_scan/config_115.h5", "desc": "17 emitters, 0.31 p/ms"},
        {"id": "scan_config_117", "path": "D:/TSRD/scan/test_scan/config_117.h5", "desc": "14 emitters, 0.91 p/ms"},
        {"id": "scan_config_128", "path": "D:/TSRD/scan/test_scan/config_128.h5", "desc": "19 emitters, 0.95 p/ms"},
    ],
    "frequency_agile": [
        {"id": "stare_config_1", "path": "D:/TSRD/stare/test_stare/config_1.h5", "desc": "7 emitters, 3-band hopping"},
        {"id": "stare_config_100", "path": "D:/TSRD/stare/test_stare/config_100.h5", "desc": "10 emitters, 3-band hopping"},
        {"id": "stare_config_101", "path": "D:/TSRD/stare/test_stare/config_101.h5", "desc": "15 emitters, 4-band hopping"},
        {"id": "stare_config_102", "path": "D:/TSRD/stare/test_stare/config_102.h5", "desc": "8 emitters, 4-band hopping"},
    ],
}


class StressSuiteEvaluator:
    """Orchestrates comprehensive stress testing and per-scenario diagnostics."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        env_config: dict[str, Any],
        model_config: dict[str, Any],
        device: str = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.env_config = copy.deepcopy(env_config)
        self.model_config = copy.deepcopy(model_config)
        self.device = torch.device(device)
        self.n_bands = int(self.env_config.get("n_bands", CANONICAL_N_BANDS))
        self.n_modes = int(self.env_config.get("n_modes", CANONICAL_N_MODES))

        # Load frozen DRQN
        logger.info("Loading frozen checkpoint: %s", self.checkpoint_path)
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        drqn_cfg = self.model_config.get("drqn_scheduler", {})
        self.drqn = DRQNScheduler(
            obs_dim=int(self.env_config.get("obs_dim", self.n_bands * 10)),
            n_bands=self.n_bands,
            n_modes=self.n_modes,
            lstm_hidden=int(drqn_cfg.get("lstm_hidden", 256)),
            lstm_layers=int(drqn_cfg.get("lstm_layers", 2)),
        ).to(self.device)
        self.drqn.load_state_dict(ckpt["state_dict"])
        self.drqn.eval()

    def evaluate_scenario(
        self,
        scenario_info: dict[str, Any],
        policy_name: str,
        n_steps: int = 1000,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Run one policy on one scenario for n_steps and return fine-grained telemetry."""
        path = Path(scenario_info["path"])
        records = load_h5_records(path, max_pulses=50000)
        if not records:
            raise ValueError(f"No records loaded from {path}")

        # Deterministic env setup (semantic memory isolated)
        val_env_cfg = copy.deepcopy(self.env_config)
        val_env_cfg["semantic_memory_enabled"] = False
        val_env_cfg["dwell_modes"] = self.model_config.get("dwell_modes")
        val_env_cfg["reward"] = self.model_config.get("reward")

        env = CognitiveRFScanEnv(
            val_env_cfg,
            records=records,
            seed=seed,
            semantic_memory_path=":memory:",
        )
        obs, _ = env.reset()

        moe_cfg = self.model_config.get("smartscan_moe", {})
        agent = build_baseline(
            policy_name,
            n_bands=self.n_bands,
            n_modes=self.n_modes,
            drqn=self.drqn,
            config=moe_cfg,
            seed=seed,
            device=str(self.device),
        )
        if hasattr(agent, "reset"):
            agent.reset()

        hidden = None
        if hasattr(agent, "init_hidden"):
            hidden = agent.init_hidden(1, self.device)

        ep_hits = 0
        ep_reward = 0.0
        band_counts = np.zeros(self.n_bands, dtype=int)
        mode_counts = np.zeros(self.n_modes, dtype=int)
        action_counts = np.zeros(self.n_bands * self.n_modes, dtype=int)

        empty_escape_opps = 0
        empty_escapes = 0
        consecutive_empty = 0
        last_band = -1

        # Detailed autonomy / reason tracking
        reasons_count: dict[str, int] = {}
        fallbacks_list: list[float] = []

        for step in range(n_steps):
            attr = None
            if hasattr(agent, "select_action"):
                if hasattr(agent, "set_periodic_urgency_vector") and getattr(env, "belief", None) is not None:
                    agent.set_periodic_urgency_vector(env.belief.periodic_urgency)
                action, hidden, attr = agent.select_action(obs, hidden)
                if attr is not None:
                    fb = float(attr.get("fallback_triggered", 0.0))
                    fallbacks_list.append(fb)
                    r = str(attr.get("reason", "unknown"))
                    reasons_count[r] = reasons_count.get(r, 0) + 1
            elif hasattr(agent, "act"):
                action, attr = agent.act(obs)
            else:
                action = agent.step(obs)

            action = int(action)
            band = int(action // self.n_modes)
            mode = int(action % self.n_modes)

            band_counts[band] += 1
            mode_counts[mode] += 1
            action_counts[action] += 1

            if last_band >= 0:
                if consecutive_empty >= 2:
                    empty_escape_opps += 1
                    if band != last_band:
                        empty_escapes += 1

            obs, reward, term, trunc, info = env.step(action)
            hit = bool(info.get("hit", False))
            ep_reward += float(reward)
            ep_hits += int(hit)

            if not hit:
                if last_band == band or last_band == -1:
                    consecutive_empty += 1
                else:
                    consecutive_empty = 1
            else:
                consecutive_empty = 0
            last_band = band

            if hasattr(agent, "update_result"):
                agent.update_result(hit, band)
            if hasattr(agent, "update"):
                agent.update(action)

            if term or trunc:
                break

        steps_done = max(1, step + 1)
        fom = env.get_fom()
        intercept_rate = float(ep_hits / steps_done)
        distinct_bands = int(np.count_nonzero(band_counts))
        empty_escape_rate = float(empty_escapes / max(1, empty_escape_opps)) if empty_escape_opps > 0 else 1.0

        fb_rate_pct = float(np.mean(fallbacks_list) * 100.0) if fallbacks_list else None
        drqn_primary_pct = float(100.0 - fb_rate_pct) if fb_rate_pct is not None else None

        mode_dist = {
            DWELL_MODES[m]: float(mode_counts[m] / steps_done * 100.0)
            for m in range(self.n_modes)
        }

        return {
            "scenario_id": scenario_info["id"],
            "category": scenario_info.get("category", "general"),
            "desc": scenario_info.get("desc", ""),
            "policy": policy_name,
            "seed": seed,
            "steps": steps_done,
            "hits": ep_hits,
            "intercept_rate": intercept_rate,
            "discovery_rate": float(fom.get("discovery_rate", 0.0) or 0.0),
            "coverage": float(fom.get("band_selection_coverage", 0.0) or 0.0),
            "distinct_bands": distinct_bands,
            "empty_escape_rate": empty_escape_rate,
            "pfa": float(fom.get("Pfa", 0.0) or 0.0),
            "avg_timing_error_us": float(fom.get("avg_intercept_time_error_us", 0.0) or 0.0),
            "total_reward": ep_reward,
            "fallback_rate_pct": fb_rate_pct,
            "drqn_primary_rate_pct": drqn_primary_pct,
            "reasons": reasons_count,
            "mode_distribution_pct": mode_dist,
        }

    def run_stress_battery(
        self,
        categories: list[str] | None = None,
        policies: list[str] | None = None,
        seeds: list[int] | None = None,
        n_steps: int = 1000,
    ) -> dict[str, Any]:
        """Execute the full stress suite across selected categories and policies."""
        target_cats = categories or list(CURATED_STRESS_SCENARIOS.keys())
        target_policies = policies or ["highest_occupancy", "highest_uncertainty", "round_robin", "drqn", "full_moe"]
        eval_seeds = seeds or [42]

        all_results: list[dict[str, Any]] = []
        t0 = time.time()

        for cat in target_cats:
            scenarios = CURATED_STRESS_SCENARIOS.get(cat, [])
            logger.info("=== Running Stress Category: %s (%d scenarios) ===", cat.upper(), len(scenarios))
            for scen in scenarios:
                scen_dict = {**scen, "category": cat}
                for seed in eval_seeds:
                    for pol in target_policies:
                        res = self.evaluate_scenario(scen_dict, policy_name=pol, n_steps=n_steps, seed=seed)
                        all_results.append(res)

        elapsed = time.time() - t0
        logger.info("Stress battery completed in %.1f seconds (%d evaluations total).", elapsed, len(all_results))
        return {
            "elapsed_seconds": elapsed,
            "checkpoint": str(self.checkpoint_path),
            "categories": target_cats,
            "policies": target_policies,
            "seeds": eval_seeds,
            "results": all_results,
        }

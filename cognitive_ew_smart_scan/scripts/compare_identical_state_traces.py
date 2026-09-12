"""Identical-State Evaluation Trace Logger.

Executes baseline and candidate models on identical scenario seeds and states,
recording per-step belief, top-5 Q-values, chosen band/mode, and ground-truth emitter activity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import torch

from cognitive_ew_smart_scan.src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from cognitive_ew_smart_scan.src.environment.scenario_generator import load_h5_records
from cognitive_ew_smart_scan.src.models.drqn_scheduler import DRQNScheduler


def collect_identical_state_trace(
    model: DRQNScheduler,
    scenario_h5_path: str | Path,
    n_steps: int = 50,
    seed: int = 42,
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """Run an evaluation episode and log step-by-step decision telemetry."""
    recs = load_h5_records(scenario_h5_path)
    env = CognitiveRFScanEnv(
        config={
            "n_bands": 36,
            "dwell_modes": 5,
            "features_per_band": 10,
            "ema_alpha": 0.30,
            "ema_alpha_miss_confirmed": 0.20,
            "semantic_memory_enabled": False,
        },
        records=recs,
        seed=seed,
    )
    obs, _ = env.reset(seed=seed)
    hx = None
    dev = torch.device(device)
    model.eval()

    trace: List[Dict[str, Any]] = []

    for step in range(n_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=dev).view(1, 1, -1)
        with torch.no_grad():
            q_out, aux, hx = model(obs_t, hx)
            q_flat = q_out.squeeze(0).squeeze(0).cpu().numpy()
            act = int(np.argmax(q_flat))

        top5_indices = np.argsort(q_flat)[-5:][::-1]
        top5 = [
            {
                "action": int(idx),
                "band": int(idx // 5),
                "mode": int(idx % 5),
                "q_value": float(q_flat[idx]),
            }
            for idx in top5_indices
        ]

        # Extract features for selected band
        b_sel = act // 5
        m_sel = act % 5
        obs_band = obs.reshape(36, 10)[b_sel]
        belief_features = {
            "occupancy": float(obs_band[0]),
            "det_rate": float(obs_band[1]),
            "uncertainty": float(obs_band[3]),
            "revisit_age": float(obs_band[4]),
            "priority": float(obs_band[9]),
        }

        obs, r, term, trunc, info = env.step(act)
        hit = bool(info.get("hit", False))

        h_norm = float(torch.norm(hx[0]).item()) if hx is not None else 0.0

        trace.append({
            "step": step,
            "selected_action": act,
            "selected_band": b_sel,
            "selected_mode": m_sel,
            "hit": hit,
            "reward": float(r),
            "hidden_norm": h_norm,
            "belief_features": belief_features,
            "top5_q": top5,
            "q_selected": float(q_flat[act]),
            "q_mean": float(np.mean(q_flat)),
        })

        if term or trunc:
            break

    return trace

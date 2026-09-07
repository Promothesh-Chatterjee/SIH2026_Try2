"""Canonical fixed evaluation batch for DRQN Q-diagnostics.

Persists a fixed, reproducible evaluation tensor [N, T, obs_dim] to disk at
`checkpoints/scheduler_clean_staged/fixed_eval_batch.pt`.

Batch composition:
  - Sequence 0: Canonical reconstructed 'stuck state' (16 steps where Band 4 is
    dwelled on continuously, leaving the other 35 bands unvisited with normalized
    age = 1.0, prio >= 0.4).
  - Sequences 1..15: Diverse validation sequences from real TSRD val_stare scenarios.

Provides `evaluate_q_diagnostics()` to track Q-margin, spread, and argmax on
the trapped state.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ..contracts import CANONICAL_N_BANDS, CANONICAL_N_MODES
from ..environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from ..environment.scenario_generator import load_h5_records

logger = logging.getLogger(__name__)

CANONICAL_EVAL_BATCH_PATH = Path("checkpoints/scheduler_clean_staged/fixed_eval_batch.pt")


def generate_stuck_state_sequence(seq_len: int = 16, stuck_band: int = 4) -> np.ndarray:
    """Generate a synthetic 16-step sequence simulating the 2-band / B4 collapse.

    At each step:
      - Band 4 has been repeatedly dwelled on: revisit age = 0.0, miss_rate = 1.0.
      - Other 35 bands have not been visited for 50+ dwells:
        revisit age = 1.0 (max), uncertainty = 0.8, priority = 0.55+.
    """
    obs_seq = np.zeros((seq_len, CANONICAL_N_BANDS * 10), dtype=np.float32)
    for t in range(seq_len):
        # Progressively advance staleness up to 1.0
        stale_age = min(1.0, float(t + 35) / 50.0)
        for b in range(CANONICAL_N_BANDS):
            offset = b * 10
            if b == stuck_band:
                obs_seq[t, offset + 0] = 0.0   # occupancy
                obs_seq[t, offset + 1] = 0.0   # det_rate
                obs_seq[t, offset + 2] = 1.0   # miss_rate
                obs_seq[t, offset + 3] = 0.1   # uncertainty
                obs_seq[t, offset + 4] = 0.0   # age
                obs_seq[t, offset + 9] = 0.05  # priority
            else:
                obs_seq[t, offset + 0] = 0.0   # occupancy
                obs_seq[t, offset + 1] = 0.0   # det_rate
                obs_seq[t, offset + 2] = 0.5   # miss_rate
                obs_seq[t, offset + 3] = 0.7   # uncertainty
                obs_seq[t, offset + 4] = stale_age  # age (stale)
                obs_seq[t, offset + 9] = 0.4 * stale_age + 0.14  # priority
    return obs_seq


def build_canonical_eval_batch(
    data_dir: str = "D:/TSRD",
    n_sequences: int = 16,
    seq_len: int = 16,
    seed: int = 42,
) -> torch.Tensor:
    """Build canonical evaluation batch [n_sequences, seq_len, obs_dim]."""
    rng = np.random.RandomState(seed)
    batch_list: list[np.ndarray] = []

    # 1. First sequence is the canonical stuck state
    stuck_seq = generate_stuck_state_sequence(seq_len=seq_len, stuck_band=4)
    batch_list.append(stuck_seq)

    # 2. Remaining 15 sequences sampled from diverse val_stare scenarios
    val_dir = Path(data_dir) / "stare" / "val_stare"
    val_files = sorted(val_dir.glob("*.h5")) if val_dir.exists() else []

    env_cfg = {
        "n_bands": CANONICAL_N_BANDS,
        "n_modes": CANONICAL_N_MODES,
        "obs_dim": CANONICAL_N_BANDS * 10,
        "semantic_memory_path": ":memory:",
    }

    scen_idx = 0
    while len(batch_list) < n_sequences:
        if val_files:
            h5_path = val_files[scen_idx % len(val_files)]
            scen_idx += 1
            try:
                records = load_h5_records(h5_path)
                env = CognitiveRFScanEnv(env_cfg, records=records, seed=seed + len(batch_list))
                obs, _ = env.reset()
                seq_obs = [obs]
                for _ in range(seq_len - 1):
                    # Random exploratory actions to populate realistic belief states
                    act = int(rng.randint(0, CANONICAL_N_BANDS * CANONICAL_N_MODES))
                    obs, _, term, trunc, _ = env.step(act)
                    seq_obs.append(obs)
                    if term or trunc:
                        break
                while len(seq_obs) < seq_len:
                    seq_obs.append(seq_obs[-1])
                batch_list.append(np.array(seq_obs[:seq_len], dtype=np.float32))
                continue
            except Exception as exc:
                logger.warning("Could not sample from %s: %s", h5_path, exc)

        # Fallback if no files: diverse random states
        fallback_seq = rng.uniform(0.0, 1.0, size=(seq_len, CANONICAL_N_BANDS * 10)).astype(np.float32)
        batch_list.append(fallback_seq)

    tensor_batch = torch.tensor(np.stack(batch_list[:n_sequences], axis=0), dtype=torch.float32)
    return tensor_batch


def get_or_create_fixed_eval_batch(
    cache_path: Path | str = CANONICAL_EVAL_BATCH_PATH,
    data_dir: str = "D:/TSRD",
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Load canonical evaluation batch from disk, or create and persist if missing."""
    p = Path(cache_path)
    if p.exists():
        try:
            tensor = torch.load(p, map_location=device)
            if tensor.dim() == 3 and tensor.size(0) >= 1 and tensor.size(-1) == CANONICAL_N_BANDS * 10:
                return tensor.to(device)
        except Exception as exc:
            logger.warning("Failed to load cached eval batch %s: %s — recreating", p, exc)

    p.parent.mkdir(parents=True, exist_ok=True)
    tensor = build_canonical_eval_batch(data_dir=data_dir)
    torch.save(tensor, p)
    logger.info("Persisted canonical fixed evaluation batch to %s [shape: %s]", p, list(tensor.shape))
    return tensor.to(device)


def evaluate_q_diagnostics(
    model: torch.nn.Module,
    eval_batch: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    """Compute Q-margin, Q-spread, and stuck-state argmax diagnostics on the fixed eval batch.

    Args:
        model: DRQNScheduler online network.
        eval_batch: Tensor of shape [N, T, obs_dim].
        device: Target execution device.

    Returns:
        Dict with keys:
          - q_margin_mean: Average top1 - top2 Q-margin across all sequences.
          - q_spread_mean: Average max(Q) - min(Q) across all sequences.
          - stuck_state_top1_band: Argmax band on the last step of the stuck sequence.
          - stuck_state_top2_band: Second-choice band on the last step of the stuck sequence.
          - stuck_state_margin: Top1 - top2 margin on the stuck sequence.
          - stuck_state_top1_mode: Selected dwell mode on the stuck sequence.
    """
    model.eval()
    t_batch = eval_batch.to(device)
    with torch.inference_mode():
        out = model(t_batch)
        q_vals = out[0] if isinstance(out, (tuple, list)) else out  # [N, T, n_actions]
        # Evaluate metrics on the last timestep of each sequence
        last_q = q_vals[:, -1, :]  # [N, n_actions]
        top2 = last_q.topk(2, dim=-1).values
        margins = top2[:, 0] - top2[:, 1]
        spreads = last_q.max(dim=-1).values - last_q.min(dim=-1).values

        # Stuck state is sequence 0
        stuck_q = last_q[0]  # [n_actions]
        stuck_top2_idx = stuck_q.topk(2, dim=-1).indices
        stuck_top2_val = stuck_q.topk(2, dim=-1).values
        n_modes = CANONICAL_N_MODES
        top1_act = int(stuck_top2_idx[0].item())
        top2_act = int(stuck_top2_idx[1].item())

        stuck_top1_band = top1_act // n_modes
        stuck_top1_mode = top1_act % n_modes
        stuck_top2_band = top2_act // n_modes
        stuck_margin = float((stuck_top2_val[0] - stuck_top2_val[1]).item())

    return {
        "q_margin_mean": float(margins.mean().item()),
        "q_spread_mean": float(spreads.mean().item()),
        "stuck_state_top1_band": stuck_top1_band,
        "stuck_state_top1_mode": stuck_top1_mode,
        "stuck_state_top2_band": stuck_top2_band,
        "stuck_state_margin": stuck_margin,
    }

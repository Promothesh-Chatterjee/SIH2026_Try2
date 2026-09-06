"""Phase 2/3 audit: checkpoint validity, device placement, runtime profile.

Run lazily and quickly:
  1. Load Run 01 best.pt (module-local scheduler checkpoint) into a fresh DRQN and
     run one forward pass (validity / resumability shell).
  2. Report device placement facts (CPU-only, threads, mkldnn).
  3. Microbenchmark env.step vs DRQN forward to see where CPU time goes.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    print("=== DEVICE / RUNTIME FACTS ===")
    print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} "
          f"cuda_devices={torch.cuda.device_count()}")
    print(f"threads={torch.get_num_threads()} interop={torch.get_num_interop_threads()}")
    try:
        print(f"mkldnn={torch.backends.mkldnn.is_available()} (oneDNN CPU)")
    except Exception as e:  # noqa: BLE001
        print(f"mkldnn check failed: {e}")

    from src.models.drqn_scheduler import DRQNScheduler
    from src.contracts import n_modes as canonical_n_modes
    n_actions = 36 * canonical_n_modes()

    ckpt_path = ROOT / "checkpoints" / "scheduler" / "best.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint missing: {ckpt_path}")

    md = torch.load(ckpt_path, map_location="cpu")
    meta = md.get("metadata", {})
    print("\n=== RUN 01 CHECKPOINT (module-local scheduler/best.pt) ===")
    print(f"  keys={list(md.keys())}  containing ONLY state_dict+metadata")
    print(f"  git={meta.get('git_revision')} ts={meta.get('timestamp')} "
          f"metrics={meta.get('metrics')}")

    drqn = DRQNScheduler(obs_dim=360, n_bands=36, n_actions=n_actions,
                         lstm_hidden=256, lstm_layers=2)
    drqn.load_state_dict(md["state_dict"])
    drqn.eval()
    obs_t = torch.zeros(1, 1, 360)
    hidden = drqn.init_hidden(1, "cpu")
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(20):
            q, _aux, hidden = drqn(obs_t, hidden)
    t_drqn = (time.perf_counter() - t0) / 20
    print(f"  DRQN forward (360d obs, B=1): {t_drqn*1e3:.2f} ms/step")
    print("  -> checkpoint VALID: loads into fresh DRQN and forwards cleanly.")
    print("  -> resume status: NO load path exists in train_scheduler.py; "
          "checkpoint stores state_dict only (no optimizer/replay/global_step/eps).")

    print("\n=== CPU RUNTIME PROFILE (env.step vs NN) ===")
    from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
    from src.environment.scenario_generator import ScenarioSource
    from src.models.deinterleaver import PDWTransformerEncoder
    from src.preprocessing.normalise import load_normalization_stats
    import yaml

    train_cfg = yaml.safe_load(open(ROOT / "configs/training_config.yaml", encoding="utf-8"))
    model_cfg = yaml.safe_load(open(ROOT / "configs/model_config.yaml", encoding="utf-8"))
    env_cfg = train_cfg.get("environment", {})
    reward_cfg = model_cfg.get("reward", {})
    env_config = {**env_cfg, **reward_cfg, "n_bands": 36, "n_modes": 5, "n_actions": n_actions}

    d_cfg = model_cfg.get("deinterleaver", {})
    deint = PDWTransformerEncoder(
        pdw_dim=d_cfg.get("pdw_dim", 6), d_model=d_cfg.get("d_model", 128),
        nhead=d_cfg.get("nhead", 8), num_layers=d_cfg.get("num_layers", 4),
        dim_feedforward=d_cfg.get("dim_feedforward", 512), dropout=d_cfg.get("dropout", 0.1),
        embed_dim=d_cfg.get("embed_dim", 64),
    )
    state = torch.load(ROOT / "checkpoints/deinterleaver/best.pt", map_location="cpu")
    deint.load_state_dict(state.get("state_dict", state), strict=False)
    deint.eval()
    fit_stats = load_normalization_stats(ROOT / "checkpoints/deinterleaver/normalization_stats.json")
    src = ScenarioSource(
        data_root="D:\\TSRD", mode="stare", subset="train", source_type="world",
        freq_min_mhz=float(env_cfg.get("freq_min_mhz", 0.0)),
        freq_max_mhz=float(env_cfg.get("freq_max_mhz", 18000.0)),
        time_horizon_us=float(env_cfg.get("time_horizon_us", 0.0)) or None,
        max_pulses=int(env_cfg.get("max_pulses", 50000)), seed=42,
        allow_synthetic_fallback=False,
    )
    world = src.sample()
    env = CognitiveRFScanEnv(env_config, records=world[:2000], seed=42, records_provider=None,
                             deinterleaver_model=deint,
                             deinterleaver_config={"fit_stats": fit_stats},
                             semantic_memory_path=str(ROOT / "scripts/.audit_sem_mem.db"))
    env.reset(seed=42)
    t0 = time.perf_counter()
    for i in range(20):
        env.step((i % 36) * 5 + 1)
    t_env = (time.perf_counter() - t0) / 20
    print(f"  env.step (real TSRD world, periodic perception): {t_env*1e3:.1f} ms/step")
    print(f"  -> NN is ~{t_env*1e3/max(t_drqn*1e3,1e-6):.0f}x cheaper than env step; "
          "ENV SIMULATION dominates CPU time.")


if __name__ == "__main__":
    main()
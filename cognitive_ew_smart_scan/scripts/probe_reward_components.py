"""PHASE 1 probe: reward component decomposition on the real TSRD scheduler env.

Runs three fixed policies on an identical shared episode world:
  * uniform_random      - uniform over all 180 (band, mode) actions
  * round_robin         - sweep bands 0..35, NORMAL dwell
  * gt_oracle_ceiling   - always tune an active band in the current dwell (GT
                          clairvoyance; calibration CEILING only, never training)

The goal is to decompose episodic reward into its components and show how much
of the reward is a constant wall (dwell_cost + false_alarm penalty on search
steps) versus policy-dependent signal (hits, novel, info_gain).

This probe never learns and never feeds GT into the policy - the oracle policy
is used purely to establish an upper bound on achievable reward.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.contracts import n_modes as canonical_n_modes  # noqa: E402

N_MODES = canonical_n_modes()

from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv  # noqa: E402
from src.environment.scenario_generator import ScenarioSource  # noqa: E402
from src.models.deinterleaver import PDWTransformerEncoder  # noqa: E402
from src.preprocessing.normalise import load_normalization_stats  # noqa: E402


def main() -> None:
    train_cfg = yaml.safe_load(open(ROOT / "configs/training_config.yaml", encoding="utf-8"))
    model_cfg = yaml.safe_load(open(ROOT / "configs/model_config.yaml", encoding="utf-8"))

    env_cfg = train_cfg.get("environment", {})
    reward_cfg = model_cfg.get("reward", {})
    env_config = {**env_cfg, **reward_cfg}
    env_config.setdefault("n_bands", int(model_cfg["drqn_scheduler"].get("n_bands", 36)))
    env_config.setdefault("n_modes", int(env_cfg.get("n_modes", 5)))
    env_config.setdefault("n_actions", int(env_cfg.get("n_actions", 180)))

    # Deinterleaver (perception) - same loading logic as train_scheduler.py.
    deinterleaver_ckpt = ROOT / train_cfg.get("deinterleaver_ckpt", "checkpoints/deinterleaver/best.pt")
    norm_stats_path = ROOT / train_cfg.get("normalization_stats", "checkpoints/deinterleaver/normalization_stats.json")
    if not deinterleaver_ckpt.exists():
        raise SystemExit(f"Missing deinterleaver checkpoint: {deinterleaver_ckpt}")

    d_cfg = model_cfg.get("deinterleaver", {})
    deinterleaver = PDWTransformerEncoder(
        pdw_dim=d_cfg.get("pdw_dim", 6),
        d_model=d_cfg.get("d_model", 128),
        nhead=d_cfg.get("nhead", 8),
        num_layers=d_cfg.get("num_layers", 4),
        dim_feedforward=d_cfg.get("dim_feedforward", 512),
        dropout=d_cfg.get("dropout", 0.1),
        embed_dim=d_cfg.get("embed_dim", 64),
    )
    state = torch.load(deinterleaver_ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    deinterleaver.load_state_dict(state, strict=False)
    deinterleaver.eval()
    fit_stats = load_normalization_stats(norm_stats_path) if norm_stats_path.exists() else None
    deint_config = {"fit_stats": fit_stats} if fit_stats else {}

    # Shared world: one real TSRD STARE train file for ALL policies (noise-free
    # comparison of reward components on the exact same RF environment).
    train_source = ScenarioSource(
        data_root="D:\\TSRD",
        mode="stare",
        subset="train",
        freq_min_mhz=float(env_cfg.get("freq_min_mhz", 0.0)),
        freq_max_mhz=float(env_cfg.get("freq_max_mhz", 18000.0)),
        time_horizon_us=float(env_cfg.get("time_horizon_us", 0.0)) or None,
        max_pulses=int(env_cfg.get("max_pulses", 50000)),
        seed=42,
        source_type="world",
        allow_synthetic_fallback=False,
    )
    world = train_source.sample()
    base_dwell = float(env_cfg.get("dwell_time_us", 500.0))
    print(f"[probe] shared world: {len(world)} pulses from TSRD STARE train", flush=True)

    def make_env(seed: int) -> CognitiveRFScanEnv:
        # Unique semantic-memory DB per policy to avoid cross-policy bleed.
        db = ROOT / "scripts" / f".probe_sem_mem_{seed}.db"
        return CognitiveRFScanEnv(
            env_config,
            records=world,
            seed=seed,
            records_provider=None,
            deinterleaver_model=deinterleaver,
            deinterleaver_config={**deint_config},
            semantic_memory_path=str(db),
        )

    n_normal = 1  # NORMAL_DWELL mode index (base dwell multiplier 1.0)

    def oracle_action(env: CognitiveRFScanEnv, rng: np.random.Generator, _step: int) -> int:
        """GT clairvoyance ceiling: pick a band active in [t, t+base_dwell]."""
        t = float(env.receiver.current_time_us) if env.receiver is not None else 0.0
        hi = t + base_dwell
        bw = (env.freq_max - env.freq_min) / max(1, env.n_bands)
        for rec in env.records:
            if rec.toa_us < hi and (rec.toa_us + float(rec.pulse_width_us)) > t:
                b = int(np.clip((float(rec.frequency_mhz) - env.freq_min) / max(bw, 1e-6), 0, env.n_bands - 1))
                return b * N_MODES + n_normal
        return int(rng.integers(0, env.action_space.n))

    policies = {
        "uniform_random": lambda env, rng, step: int(rng.integers(0, env.action_space.n)),
        "round_robin": lambda env, rng, step: (step % env.n_bands) * N_MODES + n_normal,
        "gt_oracle_ceiling": oracle_action,
    }

    N_STEPS = 300
    results: dict[str, dict] = {}
    for name, policy in policies.items():
        rng = np.random.default_rng(1234 + len(results))
        env = make_env(seed=1000 + len(results))
        env.reset(seed=1000 + len(results))
        for step in range(N_STEPS):
            action = int(policy(env, rng, step))
            _obs, _rew, term, trunc, _info = env.step(action)
            if term or trunc:
                env.reset(seed=1000 + len(results))
        results[name] = env.get_fom()

    print("\n=== REWARD COMPONENT DECOMPOSITION (300 steps, shared TSRD STARE world) ===")
    keys = [
        "avg_reward", "avg_reward_hit_term", "avg_reward_novel_term",
        "avg_reward_priority_term", "avg_reward_info_gain_term",
        "avg_reward_false_alarm_penalty", "avg_reward_dwell_cost",
        "avg_reward_miss_penalty", "avg_reward_redundant_penalty",
        "n_hits", "n_false_alarms", "n_misses",
        "avg_intercept_rate", "band_selection_coverage", "discovery_rate",
        "Pd", "Pfa", "avg_intercept_time_error_us", "spectrum_active_opportunities",
    ]
    header = f"{'policy':<18}" + "".join(f"{k:>26}" for k in keys)
    print(header)
    for name, s in results.items():
        row = f"{name:<18}" + "".join(f"{s.get(k, float('nan')):>26.4f}" if isinstance(s.get(k, 0.0), float) else f"{s.get(k, 0):>26}" for k in keys)
        print(row)
    print("\nPer-step reward budget (1000-step episode, NORMAL dwell):")
    print("  dwell_cost per step        = -0.5  (w_dwell_cost=-0.001 * 500us)")
    print("  false_alarm per empty step = -0.5  (w_false_alarm=-0.5)")
    print("  -> empty search step       ~ -1.0;  LONG dwell empty ~ -1.75")
    print("  predicted ep_reward floor  ~ -945..-1000 matched Run 01's -950..-1000")
    json.dump(results, open(ROOT / "scripts" / "probe_reward_components_result.json", "w", encoding="utf-8"), indent=2)
    print("\nSaved -> scripts/probe_reward_components_result.json")


if __name__ == "__main__":
    main()
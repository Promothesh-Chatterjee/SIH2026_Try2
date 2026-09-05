"""
Phase 13 fair baseline harness: all 11 suite entries on the SAME environment.

Fairness contract enforced here:

  * receiver        — one CognitiveRFScanEnv per baseline, built from the SAME
                      config (same SieveReceiver, threshold, ibw, band layout).
  * RF world        — identical pulse records and identical env seed.
  * dwell rules     — identical mode set, multipliers and NORMAL dwell base.
  * episode duration— identical ``max_steps`` for every baseline.
  * metrics         — identical ``FiguresOfMerit`` aggregation (env.get_fom()).
  * action space    — the canonical Discrete(n_bands*n_modes); every baseline
                      emits flat actions from that space.
  * no privilege    — baselines see only the observation (+ periodic urgency for
                      the two periodic-aware entries), never ground truth.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..models.baseline_suite import BASELINE_NAMES, NN_BASELINES, build_baseline
from ..models.drqn_scheduler import DRQNScheduler

logger = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 5000


def _load_env_config(config_path: str | Path, env_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve the canonical environment config used across the suite."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    merged: dict[str, Any] = {}
    for section in ("environment", "drqn_scheduler", "reward"):
        merged[section] = {}
    drqn_cfg = raw.get("drqn_scheduler", {})
    reward_cfg = raw.get("reward", {})
    base_env = dict(raw.get("environment", {}))
    if env_cfg:
        base_env.update(env_cfg)
    n_bands = int(drqn_cfg.get("n_bands", base_env.get("n_bands", 36)))
    n_modes = int(drqn_cfg.get("n_modes", base_env.get("n_modes", 5)))
    base_env["n_bands"] = n_bands
    base_env["n_modes"] = n_modes
    base_env["n_actions"] = int(base_env.get("n_actions", n_bands * n_modes))
    merged["environment"] = {**reward_cfg, **base_env}
    merged["drqn_scheduler"] = {**drqn_cfg, "n_bands": n_bands, "n_modes": n_modes, "n_actions": n_bands * n_modes}
    merged["reward"] = reward_cfg
    return merged


def run_baseline_episode(
    baseline,
    env_config: dict[str, Any],
    records: list[Any],
    max_steps: int = DEFAULT_MAX_STEPS,
    seed: int = 42,
) -> dict[str, float]:
    """Run one baseline through a full episode on a FRESH identical env.

    Returns this baseline's FiguresOfMerit summary (Pd, Pfa, avg_reward, ...).
    """
    from ..environment.cognitive_rf_scan_env import CognitiveRFScanEnv

    env = CognitiveRFScanEnv(env_config, records=records, seed=seed)
    obs, _ = env.reset()
    steps = 0
    try:
        baseline.reset()
    except AttributeError:
        pass
    while steps < max_steps:
        # Periodic-aware baselines consume the belief's periodic urgency — an
        # observable-history-derived signal the env itself computed, not GT.
        if hasattr(baseline, "set_periodic_urgency_vector"):
            try:
                urgency = np.asarray(getattr(env.belief, "periodic_urgency", np.zeros(env.n_bands, dtype=np.float32)), dtype=np.float32)
            except AttributeError:
                urgency = np.zeros(env.n_bands, dtype=np.float32)
            baseline.set_periodic_urgency_vector(urgency)
        if hasattr(baseline, "act"):
            action, _info = baseline.act(obs)
        else:
            action = int(baseline.step(obs))
        obs, _reward, terminated, truncated, _info = env.step(int(action))
        if hasattr(baseline, "update"):
            try:
                baseline.update(int(action))
            except Exception:
                pass
        if bool(terminated or truncated):
            break
        steps += 1
    fom = env.get_fom()
    return {f"baseline_{k}": float(v) for k, v in fom.items()}


def run_baseline_suite(
    config_path: str | Path,
    records: list[Any],
    max_steps: int = DEFAULT_MAX_STEPS,
    seed: int = 42,
    device: str = "cpu",
    drqn: DRQNScheduler | None = None,
    names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run the full (or requested subset of the) 11-baseline suite.

    Every entry gets an identical fresh env instance (same config, same records,
    same seed, same episode length). Returns one row per baseline:
    ``{"baseline": name, **baseline_<FoM summary keys>}``.
    """
    from ..environment.scenario_generator import load_h5_records  # noqa: F401  (kept for extension)
    from ..contracts import n_actions_for

    env_config = _load_env_config(config_path)
    merged = {**env_config["environment"]}
    n_bands = int(merged.get("n_bands", 36))
    n_modes = int(merged.get("n_modes", 5))
    n_actions = n_actions_for(n_bands, n_modes)

    rows: list[dict[str, Any]] = []
    selected = [str(n).lower() for n in (names or BASELINE_NAMES)]
    for name in BASELINE_NAMES:
        if name not in selected:
            continue
        baseline = build_baseline(
            name,
            n_bands=n_bands,
            n_modes=n_modes,
            seed=seed,
            device=device,
            drqn=drqn,
            config=env_config["drqn_scheduler"],
        )
        row: dict[str, Any] = {"baseline": name}
        row.update(run_baseline_episode(baseline, merged, records, max_steps=max_steps, seed=seed))
        # Audit check: the action space must be identical for every entry.
        row["n_actions"] = int(baseline.n_actions if hasattr(baseline, "n_actions") else n_actions)
        if int(row["n_actions"]) != n_actions:
            logger.warning("Baseline %s operates on a different action space (%s != %s)", name, row["n_actions"], n_actions)
        logger.info("Baseline %-18s Pd=%.3f Pfa=%.3f reward=%.4f", name, row.get("baseline_Pd", 0.0), row.get("baseline_Pfa", 0.0), row.get("baseline_avg_reward", 0.0))
        rows.append(row)
    return rows


def main() -> None:
    """CLI: run the full baseline suite over a pulse-train file or h5."""
    import argparse
    import json
    import time

    from ..environment.scenario_generator import load_h5_records

    parser = argparse.ArgumentParser(description="Run the 11-baseline fair suite")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--input", type=str, required=True, help=".h5 pulse train file")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    records = load_h5_records(Path(args.input))
    if not records:
        raise SystemExit(f"No records could be loaded from {args.input}")
    t0 = time.perf_counter()
    rows = run_baseline_suite(args.config, records, max_steps=args.max_steps, seed=args.seed)
    elapsed = (time.perf_counter() - t0) / 60.0
    for row in rows:
        print(f"{row['baseline']:<22} Pd={row.get('baseline_Pd', float('nan')):.3f}  "
              f"Pfa={row.get('baseline_Pfa', float('nan')):.3f}  "
              f"reward={row.get('baseline_avg_reward', float('nan')):.4f}")
    print(f"Suite done in {elapsed:.1f} min ({len(rows)} entries)")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
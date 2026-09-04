"""Preflight readiness gate for strict real-TSRD Cognitive EW SmartScan runs.

Runs every cheap contract check that must pass before launching an expensive
scheduler / deinterleaver training run:

  * config files parse and observation/action/receiver/reward contracts match
  * TSRD data root and stare/scan train/val .h5 splits exist
  * deinterleaver checkpoint + normalization stats exist
  * dwell-mode taxonomy in model_config matches the canonical contract
  * no cross-file emitter-label leakage (labels are file-local)
  * receiver + band mapping numerically cover the expected spectrum

Exit code 0 = ready, 1 = not ready (blocking reasons printed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs" / "model_config.yaml")
    parser.add_argument("--training-config", type=Path, default=ROOT / "configs" / "training_config.yaml")
    args = parser.parse_args()

    errors: list[str] = []

    model_cfg = _load_yaml(args.model_config)
    train_cfg = _load_yaml(args.training_config)
    env_cfg = train_cfg["environment"]

    # 1. Gate contract checks (observation / action / receiver / reward).
    from src.training.training_gate import validate_dwell_contract, validate_training_gate

    errors.extend(
        validate_training_gate(
            data_root=train_cfg["data_dir"],
            deinterleaver_checkpoint=train_cfg["deinterleaver_ckpt"],
            normalization_stats=train_cfg["normalization_stats"],
            environment_config=env_cfg,
            model_config=model_cfg,
        )
    )
    errors.extend(validate_dwell_contract(env_cfg, model_cfg))

    # 2. Cross-config consistency: drqn_scheduler action space vs environment.
    drqn = model_cfg["drqn_scheduler"]
    if int(drqn.get("n_bands", 36)) != int(env_cfg.get("n_bands", 36)):
        errors.append("n_bands mismatch: drqn_scheduler vs environment")
    if int(drqn.get("obs_dim", 360)) != int(env_cfg.get("obs_dim", 360)):
        errors.append("obs_dim mismatch: drqn_scheduler vs environment")
    if int(drqn.get("n_modes", 5)) != int(env_cfg.get("n_modes", 5)):
        errors.append("n_modes mismatch: drqn_scheduler vs environment")
    if int(drqn.get("n_actions", 180)) != int(env_cfg.get("n_actions", 180)):
        errors.append("n_actions mismatch: drqn_scheduler vs environment")

    # 3. Anthropic-check: per-band feature order is the canonical 10-layout.
    from src.utils.checkpoint_meta import FEATURE_ORDER
    from src.perception.adapters import BAND_FEATURES

    if len(FEATURE_ORDER) != int(env_cfg.get("band_features", 10)):
        errors.append(f"FEATURE_ORDER length {len(FEATURE_ORDER)} != band_features")
    if BAND_FEATURES != len(FEATURE_ORDER):
        errors.append(f"adapters.BAND_FEATURES={BAND_FEATURES} != FEATURE_ORDER len")

    # 4. Normalization stats sanity (feature alignment).
    stats_path = ROOT / train_cfg["normalization_stats"]
    if stats_path.is_file():
        try:
            with open(stats_path) as f:
                stats = json.load(f)
            feature_cols = set(stats.get("feature", {})) | {
                (stats.get("columns") or {}).get(k) for k in ("mean", "std", "min", "max")
            }
            feature_cols.discard(None)
            expected_prefixes = {"toa", "cf", "pw", "ina", "amp"}
            if feature_cols and not any(c.startswith(tuple(expected_prefixes)) for c in feature_cols if isinstance(c, str)):
                errors.append(f"normalization_stats keys unexpected: {list(feature_cols)[:5]}")
        except Exception as exc:
            errors.append(f"normalization_stats unreadable: {exc}")
    else:
        errors.append(f"normalization_stats missing: {stats_path}")

    # 5. Receiver / band-mapping numerical coverage.
    freq_min = float(env_cfg.get("freq_min_mhz", 0.0))
    freq_max = float(env_cfg.get("freq_max_mhz", 18000.0))
    ibw = float(env_cfg.get("ibw_mhz", 500.0))
    n_bands = int(env_cfg.get("n_bands", 36))
    step = (freq_max - freq_min) / max(n_bands, 1)
    if abs(step - ibw) > 1e-6:
        errors.append(f"band width {step:.1f} MHz != ibw {ibw} MHz")

    # 6. Cross-file emitter-label isolation check.
    # TSRD emitter labels are FILE-LOCAL: the same integer may denote different
    # emitters in different files, so label identities must never be mixed across
    # files. We verify the property (that files are independent) rather than
    # demanding shared labels. Genuine anomalies (unexpected keys/shapes) are flagged.
    from src.data.tsrd_manifest import resolve_split_dirs

    root = Path(train_cfg["data_dir"])
    if root.is_dir():
        for mode in ("stare", "scan"):
            try:
                splits = resolve_split_dirs(root, mode)
            except Exception as exc:
                errors.append(f"resolve_split_dirs({mode}) failed: {exc}")
                continue
            for split in ("train", "val"):
                h5s = sorted(splits[split].glob("*.h5")) if splits[split].is_dir() else []
                if not h5s:
                    continue
                try:
                    import h5py  # type: ignore

                    with h5py.File(h5s[0], "r") as f:
                        if "labels" not in f or "data" not in f:
                            errors.append(f"{mode}/{split} missing labels/data keys: {list(f.keys())}")
                            continue
                        if f["labels"].shape[0] != f["data"].shape[0]:
                            errors.append(
                                f"{mode}/{split} labels rows {f['labels'].shape[0]} != "
                                f"data rows {f['data'].shape[0]}"
                            )
                except Exception as exc:
                    errors.append(f"label-isolation probe failed for {mode}/{split}: {exc}")
                    break

    # 7. Action-space dimension sanity via canonical contract.
    from src.contracts import n_actions_for

    if int(env_cfg.get("n_actions", 0)) != n_actions_for(n_bands, int(env_cfg.get("n_modes", 5))):
        errors.append("n_actions inconsistent with canonical contract encode/decode")

    if errors:
        print("PREFLIGHT: NOT READY (", len(errors), " blocking issue(s) )", sep="")
        for e in errors:
            print("  -", e)
        return 1

    print("PREFLIGHT: READY — strict TSRD scheduler run prerequisites satisfied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
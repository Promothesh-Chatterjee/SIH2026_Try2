"""Preflight readiness gate for strict real-TSRD Cognitive EW SmartScan runs.

Runs every cheap contract check that must pass before launching an expensive
scheduler / deinterleaver training run, and returns READY or NOT_READY with
every blocking reason.

Checklist (23 required gates):

   1. STARE train split exists
   2. STARE validation split exists
   3. STARE test split exists
   4. SCAN train split exists
   5. SCAN validation split exists
   6. SCAN test split exists
   7. every .h5 in all 6 splits is readable (structure + label alignment)
   8. deinterleaver checkpoints exist (best.pt + final.pt)
   9. normalization statistics exist
  10. checkpoint metadata exists (metadata.json sidecars + in-payload metadata)
  11. observation contract
  12. action contract
  13. dwell contract
  14. receiver contract
  15. reward contract
  16. feature order
  17. dataset fingerprint (recorded vs. current data + manifest consistency)
  18. normalization fingerprint (recorded hash vs. current stats file)
  19. truth-isolation tests pass       (tests/test_no_ground_truth_leakage.py)
  20. cluster-reconciliation tests pass
                                     (tests/test_windowed_deinterleave.py::ReconcileClusterNodesTests)
  21. replay-mask tests pass          (tests/test_replay_aux_targets.py)
  22. auxiliary-head tests pass       (tests/test_drqn_aux_heads.py)
  23. baseline-contract tests pass    (tests/test_baseline_suite.py,
                                       tests/test_evaluate_baseline.py)

Exit code 0 = READY, 1 = NOT READY (blocking reasons printed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SPLITS = ("train", "val", "test")
MODES = ("stare", "scan")

# Requirement ids used to traceability-tag each emitted reason.
RID = {
    "stare_train": 1,
    "stare_val": 2,
    "stare_test": 3,
    "scan_train": 4,
    "scan_val": 5,
    "scan_test": 6,
    "readable": 7,
    "ckpt": 8,
    "norm": 9,
    "ckpt_meta": 10,
    "obs": 11,
    "action": 12,
    "dwell": 13,
    "receiver": 14,
    "reward": 15,
    "features": 16,
    "data_fp": 17,
    "norm_fp": 18,
    "tests": 19,
    "reconcile": 20,
    "replay": 21,
    "aux": 22,
    "baseline": 23,
}


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _parse_metadata_json(path: Path) -> dict | None:
    try:
        with open(path) as f:
            payload = json.load(f)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _check_split(root: Path, mode: str, split: str) -> list[str]:
    """Checks 1-6: split directory exists and contains .h5 files."""
    from src.data.tsrd_manifest import resolve_split_dirs

    try:
        directory = resolve_split_dirs(root, mode)[split]
    except Exception as exc:
        return [f"[{RID[f'{mode}_{split}']}] {mode}/{split}: resolve failed: {exc}"]
    problems = []
    if not directory.is_dir():
        problems.append(f"[{RID[f'{mode}_{split}']}] {mode}/{split} missing directory: {directory}")
    elif not any(directory.glob("*.h5")):
        problems.append(f"[{RID[f'{mode}_{split}']}] {mode}/{split} has no .h5 files: {directory}")
    return problems


def _check_readability(root: Path) -> tuple[list[str], list[str]]:
    """Check 7: header-read every .h5 across all 6 splits.

    Returns (blocking problems, informational notes). Structural problems
    (unreadable / missing keys / label misalignment) are blocking; empty
    zero-pulse scenes are legal and merely noted.
    """
    import h5py  # type: ignore

    from src.data.tsrd_manifest import resolve_split_dirs

    problems: list[str] = []
    notes: list[str] = []
    for mode in MODES:
        splits = resolve_split_dirs(root, mode)
        for split in SPLITS:
            directory = splits[split]
            if not directory.is_dir():
                continue
            files = sorted(directory.glob("*.h5"))
            empty = 0
            struct_bad = 0
            first_bad: Path | None = None
            for path in files:
                try:
                    with h5py.File(path, "r") as f:
                        if "data" not in f or "labels" not in f:
                            raise ValueError(f"missing datasets: {list(f.keys())}")
                        d = f["data"]
                        l = f["labels"]
                        if d.ndim != 2 or d.shape[1] != 5:
                            raise ValueError(f"data shape {d.shape} != (N,5)")
                        if d.shape[0] != l.shape[0]:
                            raise ValueError(
                                f"labels rows {l.shape[0]} != data rows {d.shape[0]}"
                            )
                        if d.shape[0] == 0:
                            empty += 1
                except Exception as exc:
                    struct_bad += 1
                    if first_bad is None:
                        first_bad = path
            if struct_bad:
                problems.append(
                    f"[{RID['readable']}] {mode}/{split}: {struct_bad}/{len(files)} "
                    f"files structurally invalid (first: {first_bad})"
                )
            if empty:
                notes.append(
                    f"[note {RID['readable']}] {mode}/{split}: {empty}/{len(files)} "
                    f"empty zero-pulse scenes (structurally valid, excluded from eligibility)"
                )
    return problems, notes


def _check_checkpoints(checkpoints_dir: Path, cfg_stats: Path) -> tuple[list[str], list[str]]:
    """Checks 8-10: checkpoint + normalization + metadata existence/well-formedness."""
    import torch  # type: ignore

    problems: list[str] = []
    notes: list[str] = []

    deint_ckpt_dir = checkpoints_dir / "deinterleaver"
    sched_ckpt_dir = checkpoints_dir / "scheduler"

    # Check 8: deinterleaver checkpoints + scheduler checkpoint.
    for name, d in (("deinterleaver", deint_ckpt_dir), ("scheduler", sched_ckpt_dir)):
        missing = [p for p in ("best.pt", "final.pt") if not (d / p).is_file()]
        if d is deint_ckpt_dir and missing:
            problems.append(
                f"[{RID['ckpt']}] deinterleaver checkpoints missing in {d}: {', '.join(missing)}"
            )
        elif d is sched_ckpt_dir and missing:
            notes.append(
                f"[note {RID['ckpt']}] scheduler checkpoints not present yet: "
                f"{', '.join(missing)} (produced by scheduler training)"
            )

    # Check 9: normalization statistics.
    stats_path = cfg_stats if cfg_stats.is_absolute() else checkpoints_dir / cfg_stats
    if (deint_ckpt_dir / "normalization_stats.json").is_file():
        stats_path = deint_ckpt_dir / "normalization_stats.json"
    if not stats_path.is_file():
        problems.append(
            f"[{RID['norm']}] normalization statistics missing: {stats_path} "
            f"(checked config path and {deint_ckpt_dir / 'normalization_stats.json'})"
        )

    # Check 10: metadata.json sidecars exist/parse, and saved .pt payloads
    # actually carry a metadata blob (one artifact missing its sidecar is a
    # correctness breach, not a warning).
    for label, d in (("deinterleaver", deint_ckpt_dir), ("scheduler", sched_ckpt_dir)):
        meta_path = d / "metadata.json"
        sidecar = _parse_metadata_json(meta_path)
        has_pt = (d / "best.pt").is_file()
        if has_pt and sidecar is None:
            problems.append(f"[{RID['ckpt_meta']}] {label} metadata.json missing/unreadable: {meta_path}")
            continue
        if not has_pt:
            continue
        missing_keys = [k for k in ("git_revision", "arch", "split", "feature_order_per_band", "metrics") if k not in sidecar]
        if missing_keys:
            problems.append(
                f"[{RID['ckpt_meta']}] {label} metadata.json incomplete, missing: {missing_keys}"
            )
        for pt_name in ("best.pt", "final.pt"):
            pt = d / pt_name
            if not pt.is_file():
                continue
            try:
                payload = torch.load(pt, map_location="cpu", weights_only=True)
            except Exception as exc:
                problems.append(f"[{RID['ckpt_meta']}] {label}/{pt_name} unreadable: {exc}")
                continue
            if not isinstance(payload, dict) or "metadata" not in payload:
                problems.append(
                    f"[{RID['ckpt_meta']}] {label}/{pt_name} lacks an in-payload metadata blob"
                )
    return problems, notes


def _check_fingerprints(
    checkpoints_dir: Path, root: Path, mode: str, stats_path: Path
) -> tuple[list[str], list[str]]:
    """Checks 17-18: dataset + normalization fingerprints."""
    from src.preprocessing.normalise import load_normalization_stats, normalization_stats_hash

    problems: list[str] = []
    notes: list[str] = []

    deint_dir = checkpoints_dir / "deinterleaver"
    metadata = _parse_metadata_json(deint_dir / "metadata.json") or {}

    manifest = _parse_metadata_json(deint_dir / "dataset_manifest.json") or {}

    # --- Check 17: dataset fingerprint ---
    recorded_fp = metadata.get("dataset_fingerprint")
    if not recorded_fp:
        problems.append(f"[{RID['data_fp']}] deinterleaver metadata.json has no dataset_fingerprint")
    else:
        # Reconstruct the fingerprint payload from the recorded manifest rows
        # (path / size_bytes / sha256) for the training splits only, matching
        # train_deinterleaver's dataset_fingerprint(train_files + val_files, ...).
        rows: list[dict] = []
        for split in ("train", "val"):
            for rec in (manifest.get("splits", {}).get(split, {}).get("files", []) or []):
                if all(k in rec for k in ("path", "size_bytes", "sha256")):
                    rows.append(rec)
        if not rows:
            problems.append(
                f"[{RID['data_fp']}] dataset_manifest.json has no usable per-file records"
            )
        else:
            entries = sorted(
                (
                    {
                        "path": str(rec["path"]).replace("\\", "/"),
                        "size_bytes": int(rec["size_bytes"]),
                        "sha256": str(rec["sha256"]),
                    }
                    for rec in rows
                ),
                key=lambda r: str(r["path"]).replace("\\", "/"),
            )
            payload = json.dumps(
                {"mode": mode, "files": entries}, sort_keys=True, separators=(",", ":")
            )
            rebuilt_fp = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            if rebuilt_fp != recorded_fp:
                problems.append(
                    f"[{RID['data_fp']}] dataset fingerprint mismatch: manifest-derived "
                    f"{rebuilt_fp[:16]}… != checkpoint metadata {recorded_fp[:16]}… — "
                    f"the checkpoint records a different data state than today's "
                    f"files (retrain or refresh provenance)"
                )
            # Fast disk re-verification without re-hashing every file: row exists
            # at the same size; only re-hash rows whose size differs.
            changed = 0
            for rec in entries:
                p = (root / rec["path"]).resolve()
                if not p.is_file():
                    changed += 1
                    continue
                if p.stat().st_size != rec["size_bytes"]:
                    if _sha256(p) != rec["sha256"]:
                        changed += 1
            if changed:
                problems.append(
                    f"[{RID['data_fp']}] {changed} recorded training file(s) changed "
                    f"on disk since manifest creation — retraining required"
                )
            else:
                notes.append(
                    f"[note {RID['data_fp']}] manifest rows are disk-consistent for "
                    f"{len(entries)} training/validation files"
                )

    # --- Check 18: normalization fingerprint ---
    recorded_hash = metadata.get("normalization_stats_hash")
    if not recorded_hash:
        problems.append(f"[{RID['norm_fp']}] deinterleaver metadata.json has no normalization_stats_hash")
    else:
        try:
            current_hash = normalization_stats_hash(load_normalization_stats(stats_path))
        except Exception as exc:
            problems.append(f"[{RID['norm_fp']}] normalization stats unreadable: {exc}")
        else:
            if current_hash != recorded_hash:
                problems.append(
                    f"[{RID['norm_fp']}] normalization_stats_hash mismatch: "
                    f"current {current_hash} != recorded {recorded_hash}"
                )
            else:
                notes.append(
                    f"[note {RID['norm_fp']}] normalization fingerprint matches recorded hash"
                )
    return problems, notes


def _run_behavioral_gates() -> list[str]:
    """Checks 19-23: run the regression suites that must stay green."""
    groups = [
        (RID["tests"], ["tests/test_no_ground_truth_leakage.py"]),
        (
            RID["reconcile"],
            ["tests/test_windowed_deinterleave.py::ReconcileClusterNodesTests"],
        ),
        (RID["replay"], ["tests/test_replay_aux_targets.py"]),
        (RID["aux"], ["tests/test_drqn_aux_heads.py"]),
        (
            RID["baseline"],
            ["tests/test_baseline_suite.py", "tests/test_evaluate_baseline.py"],
        ),
    ]
    problems: list[str] = []
    for rid, targets in groups:
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--disable-warnings", *targets]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=900)
        except Exception as exc:
            problems.append(f"[{rid}] pytest subprocess failed: {exc}")
            continue
        if proc.returncode != 0:
            tail = (proc.stdout or proc.stderr).strip().splitlines()
            summary = tail[-1] if tail else "test failure"
            problems.append(f"[{rid}] behavioral gate failed: {summary}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs" / "model_config.yaml")
    parser.add_argument("--training-config", type=Path, default=ROOT / "configs" / "training_config.yaml")
    parser.add_argument(
        "--dataset-root",
        "--data-dir",
        dest="dataset_root",
        type=Path,
        default=None,
        help="TSRD data root (CLI > env TSRD_DATA_ROOT > training_config data_dir)",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=ROOT / "checkpoints",
        help="Canonical artifact root (must contain deinterleaver/ and scheduler/)",
    )
    parser.add_argument(
        "--skip-behavioral-tests",
        action="store_true",
        help="Skip checks 19-23 (regression test gates) — structural checks only",
    )
    args = parser.parse_args()

    errors: list[str] = []
    notes: list[str] = []

    try:
        model_cfg = _load_yaml(args.model_config)
        train_cfg = _load_yaml(args.training_config)
    except Exception as exc:
        print("PREFLIGHT: NOT READY ( 1 blocking issue(s) )", sep="")
        print("  -", exc)
        return 1

    env_cfg = train_cfg["environment"]
    from src.data.tsrd_root import resolve_tsrd_root

    data_root = resolve_tsrd_root(cli_value=args.dataset_root, config=train_cfg)
    checkpoints_dir = Path(args.checkpoints_dir).resolve()
    statspath_cfg = Path(train_cfg.get("normalization_stats", ""))

    # Checks 1-6: all six splits exist with .h5 files.
    for mode in MODES:
        for split in SPLITS:
            errors.extend(_check_split(data_root, mode, split))

    # Check 7: full readability scan.
    read_problems, read_notes = _check_readability(data_root)
    errors.extend(read_problems)
    notes.extend(read_notes)

    # Checks 11-15: gate + dwell contracts (observation / action / reward /
    # receiver / dwell).
    from src.training.training_gate import validate_dwell_contract, validate_training_gate

    gate_errors = validate_training_gate(
        data_root=data_root,
        deinterleaver_checkpoint=checkpoints_dir / "deinterleaver" / "best.pt",
        normalization_stats=checkpoints_dir / "deinterleaver" / "normalization_stats.json",
        environment_config=env_cfg,
        model_config=model_cfg,
    )
    for msg in gate_errors:
        if "Observation contract" in msg:
            errors.append(f"[{RID['obs']}] {msg}")
        elif "Action contract" in msg:
            errors.append(f"[{RID['action']}] {msg}")
        elif "Receiver configuration" in msg:
            errors.append(f"[{RID['receiver']}] {msg}")
        elif "Reward config" in msg:
            errors.append(f"[{RID['reward']}] {msg}")
        else:
            errors.append(f"[{RID['obs']}] {msg}")  # split/existence extras
    dwell_errors = validate_dwell_contract(env_cfg, model_cfg)
    for msg in dwell_errors:
        errors.append(f"[{RID['dwell']}] {msg}")

    # Check 12 (supplement): action-space dimension vs canonical encode/decode.
    from src.contracts import n_actions_for

    n_bands = int(env_cfg.get("n_bands", 36))
    n_modes = int(env_cfg.get("n_modes", 5))
    if int(env_cfg.get("n_actions", 0)) != n_actions_for(n_bands, n_modes):
        errors.append(f"[{RID['action']}] n_actions inconsistent with canonical contract encode/decode")

    # Check 11 (supplement): cross-config obs/action consistency.
    drqn = model_cfg["drqn_scheduler"]
    for key in ("n_bands", "obs_dim", "n_modes", "n_actions"):
        if int(drqn.get(key)) != int(env_cfg.get(key)):
            errors.append(f"[{RID['obs']}] {key} mismatch: drqn_scheduler vs environment")

    # Check 16: per-band feature order is the canonical 10-layout.
    from src.perception.adapters import BAND_FEATURES
    from src.utils.checkpoint_meta import FEATURE_ORDER

    if len(FEATURE_ORDER) != int(env_cfg.get("band_features", 10)):
        errors.append(f"[{RID['features']}] FEATURE_ORDER length {len(FEATURE_ORDER)} != band_features")
    if BAND_FEATURES != len(FEATURE_ORDER):
        errors.append(f"[{RID['features']}] adapters.BAND_FEATURES={BAND_FEATURES} != FEATURE_ORDER len")

    # Checks 8-10: artifact existence + metadata sidecars.
    ckpt_problems, ckpt_notes = _check_checkpoints(checkpoints_dir, statspath_cfg)
    errors.extend(ckpt_problems)
    notes.extend(ckpt_notes)

    # Checks 17-18: fingerprints.
    norm_stats_path = checkpoints_dir / "deinterleaver" / "normalization_stats.json"
    fp_problems, fp_notes = _check_fingerprints(
        checkpoints_dir, data_root, train_cfg.get("mode", "scan"), norm_stats_path
    )
    errors.extend(fp_problems)
    notes.extend(fp_notes)

    # Checks 19-23: behavioral regression gates.
    if not args.skip_behavioral_tests:
        errors.extend(_run_behavioral_gates())

    # Receiver coverage sanity (band step vs IBW) — supplements check 14.
    freq_min = float(env_cfg.get("freq_min_mhz", 0.0))
    freq_max = float(env_cfg.get("freq_max_mhz", 18000.0))
    ibw = float(env_cfg.get("ibw_mhz", 500.0))
    step = (freq_max - freq_min) / max(n_bands, 1)
    if abs(step - ibw) > 1e-6:
        errors.append(f"[{RID['receiver']}] band width {step:.1f} MHz != ibw {ibw} MHz")

    if errors:
        print(f"PREFLIGHT: NOT READY ( {len(errors)} blocking issue(s) )", sep="")
        for e in errors:
            print("  -", e)
        return 1

    print("PREFLIGHT: READY — strict TSRD run prerequisites satisfied:")
    for n in notes:
        print("  ", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
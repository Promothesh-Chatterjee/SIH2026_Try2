"""Diagnostics report for Cognitive EW SmartScan.

Prints:
- git commit
- benchmark file path
- benchmark file SHA-256
- benchmark schema version
- checkpoint path
- checkpoint SHA-256
- dataset root
- dataset fingerprint
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def get_git_commit() -> str:
    commit = os.getenv("GIT_COMMIT")
    if not commit:
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            commit = "unknown"
    return commit


def file_sha256(p: Path) -> str:
    if not p.exists():
        return "NOT_FOUND"
    return hashlib.sha256(p.read_bytes()).hexdigest()


def compute_dataset_fingerprint(tsrd_root: str, scenarios: list[str]) -> str:
    val_dir = Path(tsrd_root) / "stare" / "val_stare"
    if not val_dir.exists():
        val_dir = Path(tsrd_root) / "val"
    hasher = hashlib.sha256()
    for sid in scenarios:
        h5_f = val_dir / f"{sid}.h5"
        if h5_f.exists():
            hasher.update(hashlib.sha256(h5_f.read_bytes()).hexdigest().encode())
        else:
            hasher.update(f"synthetic_{sid}".encode())
    return hasher.hexdigest()


def main():
    bench_p = Path("reports/benchmark_results.json")
    bench_sha = file_sha256(bench_p)
    bench_schema_ver = "unknown"
    if bench_p.exists():
        try:
            b_data = json.loads(bench_p.read_text(encoding="utf-8"))
            bench_schema_ver = b_data.get("metadata", {}).get("schema_version", "unknown")
        except Exception:
            pass

    ckpt_p = Path("experiments/checkpoints/scheduler_v2_operational_candidate/checkpoint_gate_25000_frozen.pt")
    if not ckpt_p.exists():
        ckpt_p = Path("experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt")
    ckpt_sha = file_sha256(ckpt_p)

    tsrd_root = os.getenv("TSRD_DATA_ROOT", "D:/TSRD")
    scenarios = [
        "config_117", "config_119", "config_143", "config_194", "config_195",
        "config_241", "config_29", "config_42", "config_64", "config_96",
    ]
    ds_fingerprint = compute_dataset_fingerprint(tsrd_root, scenarios)

    print("=" * 70)
    print("  COGNITIVE EW SMARTSCAN — DIAGNOSTICS PROVENANCE REPORT")
    print("=" * 70)
    print(f"git commit              : {get_git_commit()}")
    print(f"benchmark file path     : {bench_p.resolve()}")
    print(f"benchmark file SHA-256  : {bench_sha}")
    print(f"benchmark schema version: {bench_schema_ver}")
    print(f"checkpoint path         : {ckpt_p.resolve()}")
    print(f"checkpoint SHA-256      : {ckpt_sha}")
    print(f"dataset root            : {tsrd_root}")
    print(f"dataset fingerprint     : {ds_fingerprint}")
    print("=" * 70)


if __name__ == "__main__":
    main()

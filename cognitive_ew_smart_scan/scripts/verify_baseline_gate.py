"""Verification Gate for Immutable Production Baseline.

Checks:
1. Checkpoint SHA-256 matches exact canonical hash.
2. Production baseline package integrity (SHA256SUMS).
3. Evaluates or verifies metrics within explicit tolerances:
   - Mean IR: 60.45% +- 0.05%
   - Agile IR: 46.70% +- 0.05%
   - Worst-case IR: 12.40% +- 0.05%
   - Pd: 99.85% +- 0.05%
   - Pfa: <= 0.0001 (distinguishing displayed 0.00 from true zero)
4. Path safety: ensures candidate output directories NEVER resolve inside production_baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_baseline_gate")

CANONICAL_CKPT_SHA256 = "7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0"
CANONICAL_METRICS = {
    "mean_ir": 60.45,
    "agile_ir": 46.70,
    "worst_case_ir": 12.40,
    "pd": 99.85,
    "pfa": 0.0,
}
TOLERANCES = {
    "mean_ir": 0.05,
    "agile_ir": 0.05,
    "worst_case_ir": 0.05,
    "pd": 0.05,
    "pfa": 0.0001,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_sha256sums(package_dir: Path) -> bool:
    sums_file = package_dir / "SHA256SUMS"
    if not sums_file.exists():
        logger.error("SHA256SUMS not found in %s", package_dir)
        return False

    all_passed = True
    with open(sums_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            expected_hash, fname = line.split(maxsplit=1)
            target = package_dir / fname
            if not target.exists():
                logger.error("Missing file listed in SHA256SUMS: %s", target)
                all_passed = False
                continue
            actual_hash = sha256_file(target)
            if actual_hash != expected_hash:
                logger.error("Hash mismatch for %s: expected %s, got %s", fname, expected_hash, actual_hash)
                all_passed = False
            else:
                logger.info("Checksum OK: %s (%s)", fname, actual_hash[:16])
    return all_passed


def verify_metrics(metrics: Dict[str, float]) -> bool:
    all_ok = True
    for key, target in CANONICAL_METRICS.items():
        val = metrics.get(key)
        if val is None:
            # Check for alternative key names
            alt_keys = {
                "mean_ir": ["drqn_mean_ir_pct", "mean_ir_pct"],
                "agile_ir": ["drqn_agile_ir_pct", "agile_ir_pct"],
                "worst_case_ir": ["drqn_worst_case_ir_pct", "worst_case_ir_pct"],
                "pd": ["pd_pct", "decision_level_pd"],
                "pfa": ["pfa"],
            }
            for alt in alt_keys.get(key, []):
                if alt in metrics:
                    val = metrics[alt]
                    break
        if val is None:
            logger.error("Metric '%s' missing from baseline data", key)
            all_ok = False
            continue

        tol = TOLERANCES[key]
        diff = abs(val - target)
        if diff > tol:
            logger.error("Metric '%s' out of tolerance: value=%.4f, expected=%.4f (diff=%.4f > tol=%.4f)", key, val, target, diff, tol)
            all_ok = False
        else:
            logger.info("Metric OK: %s = %.4f (expected %.4f +- %.4f)", key, val, target, tol)
    return all_ok


def check_path_safety(output_dir: Path, baseline_dir: Path) -> bool:
    try:
        out_res = output_dir.resolve()
        base_res = baseline_dir.resolve()
        if out_res == base_res or base_res in out_res.parents:
            logger.error("PATH SAFETY VIOLATION: Output path %s resolves inside baseline %s", out_res, base_res)
            return False
    except Exception as exc:
        logger.error("Error resolving paths: %s", exc)
        return False
    return True


def run_verification(base_dir: Path, target_output_dir: Path | None = None) -> bool:
    logger.info("================================================================================")
    logger.info("VERIFYING PRODUCTION BASELINE GATE AT: %s", base_dir)
    logger.info("================================================================================")

    if not base_dir.exists():
        logger.error("Baseline directory %s does not exist", base_dir)
        return False

    # 1. Verify Checksum Manifest
    if not verify_sha256sums(base_dir):
        logger.error("SHA256SUMS verification failed!")
        return False

    # 2. Verify Canonical Checkpoint Hash
    ckpt_path = base_dir / "checkpoint_gate_25000_frozen.pt"
    actual_ckpt_hash = sha256_file(ckpt_path)
    if actual_ckpt_hash != CANONICAL_CKPT_SHA256:
        logger.error("Canonical checkpoint hash corrupted! Got %s, expected %s", actual_ckpt_hash, CANONICAL_CKPT_SHA256)
        return False
    logger.info("Checkpoint SHA-256 confirmed bit-exact: %s", actual_ckpt_hash)

    # 3. Verify Baseline Report Metrics
    report_file = base_dir / "benchmark_v2_baseline_gate25k.json"
    if not report_file.exists():
        logger.error("Baseline report missing: %s", report_file)
        return False

    with open(report_file, "r", encoding="utf-8") as f:
        rep_data = json.load(f)

    metrics = rep_data.get("metrics", {})
    if not verify_metrics(metrics):
        logger.error("Baseline metrics verification failed!")
        return False

    # 4. Check Path Safety if target output specified
    if target_output_dir is not None:
        if not check_path_safety(target_output_dir, base_dir):
            return False
        logger.info("Path safety OK: Target output %s will not mutate baseline", target_output_dir)

    logger.info("================================================================================")
    logger.info("ALL BASELINE GATE CHECKS PASSED: IMMUTABLE PRODUCTION CHAMPION CONFIRMED")
    logger.info("================================================================================")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Production Baseline Gate")
    parser.add_argument("--baseline-dir", type=Path, default=Path("cognitive_ew_smart_scan/checkpoints/production_baseline"))
    parser.add_argument("--candidate-output-dir", type=Path, default=None)
    args = parser.parse_args()

    success = run_verification(args.baseline_dir, args.candidate_output_dir)
    sys.exit(0 if success else 1)

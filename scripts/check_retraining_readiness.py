#!/usr/bin/env python3
"""Pre-retraining readiness gate for Gate-25k -> Gate-100k TSRD continuation.

Validates all 18 scientific, cryptographic, data integrity, and contract prerequisites
using the actual continuation config (configs/training_config_resume_100k.yaml):

 1. Frozen checkpoint exists.
 2. Frozen checkpoint SHA-256 is exact (7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0).
 3. Baseline metadata SHA manifests are consistent (reusing verify_sha256sums).
 4. TSRD root resolves correctly.
 5. All six required TSRD splits exist (stare/train, stare/val, stare/test, scan/train, scan/val, scan/test).
 6. All H5 files are structurally readable.
 7. No negative/non-monotonic raw ToA ordering exists (FAIL CLOSED on negative deltas).
 8. No unreconciled dataset fingerprint drift exists (manifest-derived hash matches).
 9. Normalization hash is correct (expected_norm_hash = bacee02ac1c29428).
10. Deinterleaver checkpoint and metadata are present (best.pt, final.pt, normalization_stats.json).
11. Canonical observation/action contracts are unchanged (obs_dim=360, n_actions=180, n_bands=36, n_modes=5).
12. Reward v2 contract is unchanged.
13. Distribution-shift monitoring is enabled (monitor_only: true).
14. Continuation config has training_mode: real_tsrd.
15. Continuation config adheres to project semantic continuation contract weights_only: true.
16. Parent checkpoint is exactly the frozen 25k SHA.
17. Output directory is outside production_baseline and scheduler_v2_operational_candidate.
18. Checkpoint SHA unchanged pre- and post-readiness check (baseline strictly immutable; no training launched).

Exit Code:
  0: READY_FOR_TSRD_RETRAINING
  1: NOT_READY_FOR_TSRD_RETRAINING
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Tuple

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("check_readiness")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CANONICAL_FROZEN_SHA = "7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0"
FROZEN_CKPT_PATH = REPO_ROOT / "experiments" / "checkpoints" / "production_baseline" / "checkpoint_gate_25000_frozen.pt"
RESUME_CONFIG_PATH = REPO_ROOT / "configs" / "training_config_resume_100k.yaml"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class ReadinessChecker:
    def __init__(self, tsrd_root: str | None = None, resume_config: Path = RESUME_CONFIG_PATH):
        self.resume_config_path = Path(resume_config)
        self.cli_tsrd_root = tsrd_root
        self.blockers: List[str] = []
        self.passed_checks: List[str] = []
        self.pre_sha: str = ""

    def run_check(self, num: int, title: str, passed: bool, detail: str = ""):
        if passed:
            self.passed_checks.append(f"[{num:02d}] {title}: {detail}")
            logger.info("[CHECK %02d PASS] %s", num, title)
        else:
            msg = f"[{num:02d}] {title}: {detail}"
            self.blockers.append(msg)
            logger.error("[CHECK %02d FAIL] %s — %s", num, title, detail)

    def check_all(self) -> bool:
        try:
            self.current_head_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
            ).strip()
        except Exception:
            self.current_head_sha = "UNKNOWN"

        logger.info("================================================================================")
        logger.info("PRE-RETRAINING READINESS GATE: Gate-25k -> Gate-100k TSRD Continuation")
        logger.info("Repository Git HEAD: %s", self.current_head_sha)
        logger.info("Continuation Config: %s", self.resume_config_path)
        logger.info("================================================================================")

        # 18. Pre-check baseline hashing
        if FROZEN_CKPT_PATH.exists():
            self.pre_sha = _sha256(FROZEN_CKPT_PATH)

        # Check 1: Frozen checkpoint exists
        c1_ok = FROZEN_CKPT_PATH.is_file()
        self.run_check(1, "Frozen Checkpoint Exists", c1_ok, str(FROZEN_CKPT_PATH))
        if not c1_ok:
            return False

        # Check 2: Frozen checkpoint SHA is exact
        c2_ok = (self.pre_sha == CANONICAL_FROZEN_SHA)
        self.run_check(2, "Frozen Checkpoint Canonical SHA-256", c2_ok, f"SHA: {self.pre_sha[:16]}...")
        if not c2_ok:
            return False

        # Check 3: Baseline metadata SHA manifests are consistent (reusing verify_sha256sums)
        from scripts.verify_baseline_gate import verify_sha256sums
        package_dir = FROZEN_CKPT_PATH.parent
        sums_verified = verify_sha256sums(package_dir)
        self.run_check(3, "Baseline Manifest & SHA256SUMS Integrity", sums_verified, "All package checksums verified")

        # Load continuation config
        if not self.resume_config_path.is_file():
            self.run_check(4, "Resume Configuration Exists", False, f"Missing {self.resume_config_path}")
            return False

        with open(self.resume_config_path, "r", encoding="utf-8") as f:
            resume_cfg = yaml.safe_load(f)

        # Check 4: TSRD root resolves correctly
        from ew_core.data.tsrd_root import resolve_tsrd_root
        data_root = resolve_tsrd_root(cli_value=Path(self.cli_tsrd_root) if self.cli_tsrd_root else None, config=resume_cfg)
        c4_ok = data_root.is_dir()
        self.run_check(4, "TSRD Root Resolution", c4_ok, f"Resolved to {data_root}")
        if not c4_ok:
            return False

        # Check 5: All six required TSRD splits exist
        from ew_core.data.tsrd_manifest import resolve_split_dirs
        c5_missing = []
        for mode in ("stare", "scan"):
            try:
                splits = resolve_split_dirs(data_root, mode)
                for split in ("train", "val", "test"):
                    sp_dir = splits[split]
                    if not sp_dir.is_dir() or not any(sp_dir.glob("*.h5")):
                        c5_missing.append(f"{mode}/{split} ({sp_dir})")
            except Exception as exc:
                c5_missing.append(f"{mode}: {exc}")
        self.run_check(5, "Six Required TSRD Splits Exist", len(c5_missing) == 0,
                       f"Missing: {c5_missing}" if c5_missing else "All 6 splits populated with .h5 files")

        # Check 6: All H5 files are structurally readable
        from scripts.preflight_tsrd import _check_readability
        read_problems, read_notes = _check_readability(data_root)
        self.run_check(6, "H5 Files Structural Readability", len(read_problems) == 0,
                       f"Readability problems: {read_problems}" if read_problems else "All H5 files structurally valid")

        # Check 7: No negative/non-monotonic raw ToA ordering (FAIL CLOSED)
        from scripts.preflight_tsrd import _check_temporal_integrity
        temp_problems, temp_notes = _check_temporal_integrity(data_root)
        self.run_check(7, "Strict Raw ToA Monotonicity (Check 24)", len(temp_problems) == 0,
                       f"Temporal problems: {temp_problems}" if temp_problems else "Zero negative ToA deltas (monotonic)")

        # Check 8: No unreconciled dataset fingerprint drift exists
        checkpoints_dir = REPO_ROOT / "experiments" / "checkpoints"
        from scripts.preflight_tsrd import _check_fingerprints
        norm_stats_path = checkpoints_dir / "deinterleaver" / "normalization_stats.json"
        fp_problems, fp_notes = _check_fingerprints(checkpoints_dir, data_root, resume_cfg.get("mode", "scan"), norm_stats_path)
        c8_ok = len([p for p in fp_problems if "dataset fingerprint" in p.lower() or "changed on disk" in p.lower()]) == 0
        self.run_check(8, "Dataset Manifest Fingerprint Alignment", c8_ok,
                       "Manifest rows disk-consistent" if c8_ok else f"Fingerprint issues: {fp_problems}")

        # Check 9: Normalization stats hash exact
        expected_norm_hash = str(resume_cfg.get("expected_norm_hash", "bacee02ac1c29428"))
        from ew_core.preprocessing.normalise import load_normalization_stats, normalization_stats_hash
        try:
            curr_norm_hash = normalization_stats_hash(load_normalization_stats(norm_stats_path))
            c9_ok = (curr_norm_hash == expected_norm_hash)
            c9_detail = f"Hash={curr_norm_hash} (expected={expected_norm_hash})"
        except Exception as exc:
            c9_ok = False
            c9_detail = str(exc)
        self.run_check(9, "Normalization Stats Hash Verification", c9_ok, c9_detail)

        # Check 10: Deinterleaver checkpoint and metadata are present
        deint_dir = checkpoints_dir / "deinterleaver"
        deint_missing = [p for p in ("best.pt", "final.pt", "metadata.json", "normalization_stats.json") if not (deint_dir / p).is_file()]
        self.run_check(10, "Deinterleaver Checkpoints & Metadata", len(deint_missing) == 0,
                       f"Missing: {deint_missing}" if deint_missing else f"All artifacts present in {deint_dir}")

        # Check 11: Canonical observation/action contracts are unchanged
        env_cfg = resume_cfg.get("environment", {})
        c11_ok = (
            int(env_cfg.get("obs_dim", 0)) == 360
            and int(env_cfg.get("n_actions", 0)) == 180
            and int(env_cfg.get("n_bands", 0)) == 36
            and int(env_cfg.get("n_modes", 0)) == 5
        )
        self.run_check(11, "Canonical Architecture Contracts (360-D / 180-A)", c11_ok,
                       f"obs_dim={env_cfg.get('obs_dim')}, n_actions={env_cfg.get('n_actions')}, n_bands={env_cfg.get('n_bands')}, n_modes={env_cfg.get('n_modes')}")

        # Check 12: Reward v2 contract is unchanged
        rew_cfg = resume_cfg.get("reward", {})
        c12_ok = (rew_cfg.get("version") == "v2" and rew_cfg.get("variant") == "baseline")
        self.run_check(12, "Reward v2 Contract", c12_ok, f"version={rew_cfg.get('version')}, variant={rew_cfg.get('variant')}")

        # Check 13: Distribution-shift monitoring is enabled (monitor_only: true)
        ds_cfg = resume_cfg.get("distribution_shift", {})
        c13_ok = (
            bool(ds_cfg.get("enabled", False)) is True
            and bool(ds_cfg.get("monitor_only", False)) is True
            and bool(ds_cfg.get("auto_adjust_learning_rate", True)) is False
            and bool(ds_cfg.get("auto_adjust_replay_ratio", True)) is False
        )
        self.run_check(13, "Policy-Observation Distribution Shift Monitoring", c13_ok,
                       f"enabled={ds_cfg.get('enabled')}, monitor_only={ds_cfg.get('monitor_only')}")

        # Check 14: Continuation config is training_mode: real_tsrd
        c14_ok = (resume_cfg.get("training_mode") == "real_tsrd")
        self.run_check(14, "Real TSRD Training Mode Enforcement", c14_ok, f"training_mode={resume_cfg.get('training_mode')}")

        # Check 15: Continuation contract + fresh master-verification provenance
        sched_sec = resume_cfg.get("scheduler", {})
        weights_only_ok = bool(resume_cfg.get("weights_only", False)) and bool(sched_sec.get("weights_only", False))
        real_tsrd_ok = (resume_cfg.get("training_mode") == "real_tsrd")

        master_rpt_path = REPO_ROOT / "reports" / "master_verification_results.json"
        master_ok = False
        master_detail = ""
        if master_rpt_path.is_file():
            try:
                with open(master_rpt_path, "r", encoding="utf-8") as f:
                    master_data = json.load(f)
                rpt_passed = master_data.get("passed_gates", 0)
                rpt_failed = master_data.get("failed_gates", 0)
                rpt_total = master_data.get("total_gates", 0)
                rpt_commit = master_data.get("provenance", {}).get("verification_orchestrator_commit") or master_data.get("verification_orchestrator_commit")

                # Check gate 14 was NOT skipped
                gate14_res = [g for g in master_data.get("gate_results", []) if g.get("gate_id") == 14]
                gate14_not_skipped = len(gate14_res) > 0 and gate14_res[0].get("passed") is True and "skipped" not in str(gate14_res[0].get("detail", "")).lower()

                evidence_commit = master_data.get("provenance", {}).get("evidence_package_commit")
                commit_match = (
                    rpt_commit == self.current_head_sha
                    or (evidence_commit is not None and evidence_commit == self.current_head_sha)
                    or subprocess.run(["git", "merge-base", "--is-ancestor", rpt_commit, self.current_head_sha],
                                      cwd=str(REPO_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
                )

                master_ok = (rpt_passed == 15 and rpt_failed == 0 and rpt_total == 15 and gate14_not_skipped and commit_match)
                if not commit_match:
                    master_detail = f"Report commit {str(rpt_commit)[:8]} != HEAD {str(self.current_head_sha)[:8]}"
                elif not gate14_not_skipped:
                    master_detail = "Gate 14 smoke was skipped or failed"
                else:
                    master_detail = f"15/15 passed, Gate14 smoke verified, commit={str(rpt_commit)[:8]} (matches HEAD)"
            except Exception as exc:
                master_ok = False
                master_detail = f"Error reading master report: {exc}"
        else:
            master_detail = f"Missing {master_rpt_path}"

        c15_ok = weights_only_ok and real_tsrd_ok and master_ok
        c15_title = "Continuation Contract & Master Verification Provenance"
        c15_msg = f"weights_only: {weights_only_ok}; Master 15-Gate: {master_detail}"
        self.run_check(15, c15_title, c15_ok, c15_msg)

        # Check 16: Parent checkpoint is exactly the frozen 25k SHA
        parent_ckpt_rel = resume_cfg.get("scheduler_ckpt")
        parent_ckpt_path = REPO_ROOT / parent_ckpt_rel if parent_ckpt_rel else None
        if parent_ckpt_path and parent_ckpt_path.is_file():
            parent_sha = _sha256(parent_ckpt_path)
            c16_ok = (parent_sha == CANONICAL_FROZEN_SHA)
            c16_detail = f"Parent SHA={parent_sha[:16]}... (matches canonical 25k)" if c16_ok else f"Mismatch: {parent_sha}"
        else:
            c16_ok = False
            c16_detail = f"Parent checkpoint not found at {parent_ckpt_path}"
        self.run_check(16, "Parent Checkpoint Canonical Hash Pinning", c16_ok, c16_detail)

        # Check 17: Output directory is outside production_baseline and operational candidate
        from ew_core.utils.checkpoint_paths import SCHEDULER_DIR, resolve_checkpoint_dir
        resolved_out = resolve_checkpoint_dir(None, resume_cfg.get("output_dir"), SCHEDULER_DIR, role="scheduler").resolve()
        forbidden_roots = [
            (REPO_ROOT / "experiments" / "checkpoints" / "production_baseline").resolve(),
            (REPO_ROOT / "experiments" / "checkpoints" / "scheduler_v2_operational_candidate").resolve(),
        ]
        c17_ok = all(resolved_out != f and f not in resolved_out.parents for f in forbidden_roots)
        self.run_check(17, "Output Directory Isolation", c17_ok, f"Output directory {resolved_out} outside baseline")

        # Check 18: Checkpoint SHA unchanged pre- and post-readiness check
        post_sha = _sha256(FROZEN_CKPT_PATH)
        c18_ok = (post_sha == self.pre_sha == CANONICAL_FROZEN_SHA)
        self.run_check(18, "Frozen Baseline Checkpoint Bit-Exact Immutability", c18_ok,
                       "Pre/post SHA identical; no production continuation training executed (Gate 14 isolated smoke only)")

        # Final verdict
        all_passed = (len(self.blockers) == 0)
        logger.info("================================================================================")
        if all_passed:
            print(f"\nREADY_FOR_TSRD_RETRAINING (HEAD: {self.current_head_sha})\n")
            logger.info("READINESS GATE STATUS: READY_FOR_TSRD_RETRAINING (All 18 criteria satisfied cleanly at HEAD %s)", self.current_head_sha)
        else:
            print(f"\nNOT_READY_FOR_TSRD_RETRAINING (HEAD: {self.current_head_sha})\n")
            logger.error("READINESS GATE STATUS: NOT_READY_FOR_TSRD_RETRAINING (%d blocking issues at HEAD %s)", len(self.blockers), self.current_head_sha)
            for b in self.blockers:
                logger.error("  BLOCKER: %s", b)
        logger.info("================================================================================")
        return all_passed


def main():
    parser = argparse.ArgumentParser(description="Pre-retraining Readiness Gate")
    parser.add_argument("--tsrd-root", type=str, default="D:/TSRD", help="TSRD root dataset path")
    parser.add_argument("--resume-config", type=Path, default=RESUME_CONFIG_PATH, help="Path to continuation training YAML config")
    args = parser.parse_args()

    checker = ReadinessChecker(tsrd_root=args.tsrd_root, resume_config=args.resume_config)
    success = checker.check_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

"""Unit tests for production baseline immutability and gate verification."""

import hashlib
import json
from pathlib import Path
import unittest

from cognitive_ew_smart_scan.scripts.verify_baseline_gate import (
    CANONICAL_CKPT_SHA256,
    CANONICAL_METRICS,
    TOLERANCES,
    sha256_file,
    verify_sha256sums,
    verify_metrics,
    check_path_safety,
    run_verification,
)


class TestProductionBaselineImmutable(unittest.TestCase):
    def setUp(self):
        self.base_dir = Path("cognitive_ew_smart_scan/checkpoints/production_baseline")

    def test_checkpoint_hash_matches_canonical(self):
        ckpt_path = self.base_dir / "checkpoint_gate_25000_frozen.pt"
        self.assertTrue(ckpt_path.exists(), f"Missing baseline checkpoint at {ckpt_path}")
        h = sha256_file(ckpt_path)
        self.assertEqual(h, CANONICAL_CKPT_SHA256)

    def test_checksum_manifest(self):
        self.assertTrue(verify_sha256sums(self.base_dir))

    def test_baseline_metadata_contents(self):
        meta_file = self.base_dir / "baseline_metadata.json"
        self.assertTrue(meta_file.exists())
        with open(meta_file, "r") as f:
            meta = json.load(f)
        self.assertEqual(meta["status"], "IMMUTABLE_PRODUCTION_BASELINE")
        self.assertEqual(meta["checkpoint_sha256"], CANONICAL_CKPT_SHA256)
        self.assertEqual(meta["architecture"]["n_bands"], 36)
        self.assertEqual(meta["architecture"]["n_modes"], 5)
        self.assertEqual(meta["architecture"]["n_actions"], 180)

    def test_metrics_within_documented_tolerances(self):
        rep_file = self.base_dir / "benchmark_v2_baseline_gate25k.json"
        self.assertTrue(rep_file.exists())
        with open(rep_file, "r") as f:
            data = json.load(f)
        self.assertTrue(verify_metrics(data["metrics"]))

    def test_path_safety_rejects_baseline_mutation(self):
        # Attempting to set candidate output inside baseline must be rejected
        unsafe_path = self.base_dir / "candidate_subfolder"
        self.assertFalse(check_path_safety(unsafe_path, self.base_dir))

        # Safe separate path must be accepted
        safe_path = Path("cognitive_ew_smart_scan/checkpoints/safe_continuation_run")
        self.assertTrue(check_path_safety(safe_path, self.base_dir))

    def test_full_verification_gate(self):
        safe_output = Path("cognitive_ew_smart_scan/checkpoints/test_run_candidate")
        self.assertTrue(run_verification(self.base_dir, safe_output))


if __name__ == "__main__":
    unittest.main()

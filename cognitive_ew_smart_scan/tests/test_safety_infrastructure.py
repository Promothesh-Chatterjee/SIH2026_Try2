"""Unit tests for safety monitor, checkpoint guard, and rollback manager."""

import json
import tempfile
from pathlib import Path
import unittest
import torch

from cognitive_ew_smart_scan.src.training.safety.checkpoint_guard import CheckpointGuard
from cognitive_ew_smart_scan.src.training.safety.safety_monitor import SafetyMonitor
from cognitive_ew_smart_scan.src.training.safety.rollback_manager import RollbackManager


class TestSafetyInfrastructure(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_checkpoint_guard_rejects_forbidden_path(self):
        forbidden = self.tmp_path / "production_baseline"
        forbidden.mkdir()

        # Attempting to set output_dir to forbidden directory must raise RuntimeError
        with self.assertRaises(RuntimeError) as ctx:
            CheckpointGuard(output_dir=forbidden, forbidden_dirs=[forbidden])
        self.assertIn("CRITICAL PATH SAFETY VIOLATION", str(ctx.exception))

    def test_checkpoint_guard_atomic_save_and_hash(self):
        out_dir = self.tmp_path / "candidate_dir"
        guard = CheckpointGuard(output_dir=out_dir)

        payload = {"step": 25000, "state": "test_state"}
        saved_path, sha_hex = guard.save_checkpoint_atomic(payload, "checkpoint_test.pt")

        self.assertTrue(saved_path.exists())
        self.assertEqual(len(sha_hex), 64)
        loaded = torch.load(saved_path, weights_only=False)
        self.assertEqual(loaded["step"], 25000)

    def test_safety_monitor_tiers(self):
        monitor = SafetyMonitor(halt_ceiling_q=50.0)

        # 1. Clean healthy
        clean_action = {"safety_halt": False, "diagnostic_warning": False}
        clean_q = {"safety_halt": False, "diagnostic_warning": False}
        res_clean = monitor.check(clean_action, clean_q)
        self.assertFalse(res_clean["halt_triggered"])
        self.assertFalse(res_clean["has_warnings"])

        # 2. Diagnostic Warning
        warn_action = {
            "safety_halt": False,
            "diagnostic_warning": True,
            "warning_reasons": ["Warning top-band dominance: 65% > 60%"],
        }
        res_warn = monitor.check(warn_action, clean_q)
        self.assertFalse(res_warn["halt_triggered"])
        self.assertTrue(res_warn["has_warnings"])

        # 3. Hard Safety Halt
        halt_q = {
            "safety_halt": True,
            "diagnostic_warning": True,
            "halt_reasons": ["Hard safety halt: max_q=55.0 > 50.0"],
        }
        res_halt = monitor.check(clean_action, halt_q)
        self.assertTrue(res_halt["halt_triggered"])
        self.assertIn("Hard safety halt: max_q=55.0 > 50.0", res_halt["halt_reasons"])

    def test_rollback_manager(self):
        # Create dummy initial known-good
        init_good = self.tmp_path / "checkpoint_gate_25000_frozen.pt"
        torch.save({"step": 25000}, init_good)

        audit_log = self.tmp_path / "rollback_audit_log.json"
        rm = RollbackManager(last_known_good_ckpt=init_good, audit_log_path=audit_log)

        # Create dummy failed candidate
        failed_candidate = self.tmp_path / "checkpoint_gate_27500_candidate.pt"
        torch.save({"step": 27500}, failed_candidate)

        restored = rm.execute_rollback(
            failed_ckpt_path=failed_candidate,
            reasons=["Top-band dominance 85% >= 80%"],
            step=27500,
        )
        self.assertEqual(restored, init_good.resolve())
        self.assertTrue(audit_log.exists())

        with open(audit_log, "r") as f:
            data = json.load(f)
        self.assertEqual(len(data["rollback_events"]), 1)
        self.assertEqual(data["rollback_events"][0]["step"], 27500)

        # Candidate should have been renamed to quarantined
        quarantined = self.tmp_path / "checkpoint_gate_27500_candidate_QUARANTINED_COLLAPSE.pt"
        self.assertTrue(quarantined.exists())
        self.assertFalse(failed_candidate.exists())


if __name__ == "__main__":
    unittest.main()

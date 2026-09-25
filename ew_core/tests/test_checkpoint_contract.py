"""Tests for Phase 8 Hardened Checkpoint Contract."""

import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ew_core.utils.checkpoint_meta import (
    CheckpointMode,
    save_hardened_checkpoint,
    load_hardened_checkpoint,
    validate_checkpoint,
    quarantine_checkpoint,
    migrate_checkpoint_to_hardened,
)


class DummyModel(nn.Module):
    def __init__(self, in_features=10, out_features=5):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(x)


class TestCheckpointContract(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_weights_only_contract(self):
        model = DummyModel()
        target = DummyModel()
        opt = optim.Adam(model.parameters(), lr=1e-3)

        ckpt_path = self.dir_path / "test_weights_only.pt"
        save_hardened_checkpoint(
            path=ckpt_path,
            mode=CheckpointMode.WEIGHTS_ONLY,
            model=model,
            target_model=target,
            optimizer=opt,
            global_step=1234,
            episode=56,
            epsilon=0.42,
        )

        # Create new model instances to load into
        new_model = DummyModel()
        new_target = DummyModel()
        new_opt = optim.Adam(new_model.parameters(), lr=1e-3)

        loaded = load_hardened_checkpoint(
            path=ckpt_path,
            expected_mode=CheckpointMode.WEIGHTS_ONLY,
            model=new_model,
            target_model=new_target,
            optimizer=new_opt,
        )

        # Verify weights match
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))
        for p1, p2 in zip(target.parameters(), new_target.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

        # Verify optimizer and RNG were NOT restored and counters reset
        self.assertFalse(loaded["optimizer_restored"])
        self.assertFalse(loaded["rng_restored"])
        self.assertEqual(loaded["global_step"], 0)
        self.assertEqual(loaded["episode"], 0)
        self.assertIsNone(loaded["epsilon"])
        self.assertEqual(loaded["mode"], CheckpointMode.WEIGHTS_ONLY.value)

    def test_exact_continuation_contract(self):
        model = DummyModel()
        target = DummyModel()
        opt = optim.Adam(model.parameters(), lr=1e-3)

        # Take a step to make optimizer state non-empty
        loss = model(torch.randn(2, 10)).sum()
        loss.backward()
        opt.step()

        ckpt_path = self.dir_path / "test_exact_resume.pt"
        save_hardened_checkpoint(
            path=ckpt_path,
            mode=CheckpointMode.EXACT_CONTINUATION,
            model=model,
            target_model=target,
            optimizer=opt,
            global_step=45000,
            episode=120,
            epsilon=0.15,
        )

        new_model = DummyModel()
        new_target = DummyModel()
        new_opt = optim.Adam(new_model.parameters(), lr=1e-3)

        loaded = load_hardened_checkpoint(
            path=ckpt_path,
            expected_mode=CheckpointMode.EXACT_CONTINUATION,
            model=new_model,
            target_model=new_target,
            optimizer=new_opt,
        )

        self.assertTrue(loaded["optimizer_restored"])
        self.assertTrue(loaded["rng_restored"])
        self.assertEqual(loaded["global_step"], 45000)
        self.assertEqual(loaded["episode"], 120)
        self.assertAlmostEqual(loaded["epsilon"], 0.15)
        self.assertEqual(loaded["mode"], CheckpointMode.EXACT_CONTINUATION.value)

    def test_rng_exact_reproducibility(self):
        """Verify RNG restoration reproduces exact pseudo-random sequences."""
        model = DummyModel()
        torch.manual_seed(42)
        np.random.seed(42)

        # Generate a baseline value
        expected_torch = torch.randn(5)
        expected_np = np.random.rand(5)

        # Reset seeds and save
        torch.manual_seed(42)
        np.random.seed(42)
        ckpt_path = self.dir_path / "test_rng.pt"
        save_hardened_checkpoint(
            path=ckpt_path,
            mode=CheckpointMode.EXACT_CONTINUATION,
            model=model,
        )

        # Advance RNG states
        _ = torch.randn(100)
        _ = np.random.rand(100)

        # Load and restore RNG
        load_hardened_checkpoint(
            path=ckpt_path,
            expected_mode=CheckpointMode.EXACT_CONTINUATION,
            model=model,
            restore_rng=True,
        )

        restored_torch = torch.randn(5)
        restored_np = np.random.rand(5)

        self.assertTrue(torch.allclose(expected_torch, restored_torch))
        self.assertTrue(np.allclose(expected_np, restored_np))

    def test_validate_checkpoint_healthy_and_corrupt(self):
        model = DummyModel()
        ckpt_path = self.dir_path / "valid.pt"
        save_hardened_checkpoint(
            path=ckpt_path,
            mode=CheckpointMode.WEIGHTS_ONLY,
            model=model,
        )

        is_valid, errors = validate_checkpoint(ckpt_path, mode=CheckpointMode.WEIGHTS_ONLY, model=model)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

        # Test shape mismatch
        mismatched_model = DummyModel(in_features=20, out_features=5)
        is_valid, errors = validate_checkpoint(ckpt_path, model=mismatched_model)
        self.assertFalse(is_valid)
        self.assertTrue(any("Shape mismatch" in e for e in errors))

        # Test NaN detection
        nan_model = DummyModel()
        with torch.no_grad():
            nan_model.fc.weight[0, 0] = float("nan")
        nan_ckpt = self.dir_path / "nan.pt"
        torch.save({"state_dict": nan_model.state_dict()}, nan_ckpt)
        is_valid, errors = validate_checkpoint(nan_ckpt)
        self.assertFalse(is_valid)
        self.assertTrue(any("NaN" in e for e in errors))

    def test_quarantine_checkpoint(self):
        model = DummyModel()
        ckpt_path = self.dir_path / "bad_checkpoint.pt"
        save_hardened_checkpoint(
            path=ckpt_path,
            mode=CheckpointMode.WEIGHTS_ONLY,
            model=model,
        )

        quarantine_dir = self.dir_path / "quarantined"
        q_path = quarantine_checkpoint(
            path=ckpt_path,
            quarantine_dir=quarantine_dir,
            reason="Q-value runaway detected during validation",
        )

        self.assertFalse(ckpt_path.exists())
        self.assertTrue(q_path.exists())
        meta_sidecar = q_path.with_name(f"{q_path.name}.quarantine_meta.json")
        self.assertTrue(meta_sidecar.exists())

        import json
        with open(meta_sidecar, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["reason"], "Q-value runaway detected during validation")
        self.assertIn("sha256", data)

    def test_migrate_checkpoint_to_hardened(self):
        model = DummyModel()
        legacy_path = self.dir_path / "legacy.pt"
        torch.save({"state_dict": model.state_dict(), "global_step": 25000}, legacy_path)

        hardened_path = self.dir_path / "hardened.pt"
        migrate_checkpoint_to_hardened(
            src_path=legacy_path,
            dst_path=hardened_path,
            mode=CheckpointMode.WEIGHTS_ONLY,
        )

        self.assertTrue(hardened_path.exists())
        is_valid, errors = validate_checkpoint(hardened_path, mode=CheckpointMode.WEIGHTS_ONLY)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)


if __name__ == "__main__":
    unittest.main()

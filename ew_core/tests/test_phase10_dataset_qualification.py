"""Phase 10 — TSRD Dataset Qualification Unit Tests.

Covers:
- Streaming canonical content hash invariance to compression/metadata
- Streaming canonical content hash sensitivity to pulse/label changes
- 3-layer cross-split isolation verification (Path, Raw SHA-256, Canonical Content SHA-256)
- Near-duplicate diagnostic fingerprint behavior (non-blocking)
- Project 8-class EW taxonomy classification & unknown fallback
- Dataset role contract enforcement (train cannot be used for evaluation)
- Strict non-decreasing ToA validation (fail-closed, no silent sorting)
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from ew_core.data.tsrd_manifest import (
    PROJECT_TAXONOMY_CLASSES,
    SplitLeakageError,
    TSRDValidator,
    build_manifest,
    classify_project_taxonomy,
    compute_near_duplicate_diagnostic,
    streaming_canonical_content_sha256,
    validate_split_isolation,
)


def _make_h5(
    path: Path,
    data: np.ndarray,
    labels: np.ndarray,
    compression: str | None = None,
    compression_opts: int | None = None,
    attrs: dict | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w") as h:
        kwargs = {}
        if compression:
            kwargs["compression"] = compression
        if compression_opts is not None:
            kwargs["compression_opts"] = compression_opts

        h.create_dataset("data", data=data.astype(np.float32), **kwargs)
        h.create_dataset("labels", data=labels.astype(np.int64).reshape(-1, 1), **kwargs)
        if attrs:
            for k, v in attrs.items():
                h.attrs[k] = v
    return path


class Phase10StreamingCanonicalContentHashTests(unittest.TestCase):
    """Verify streaming canonical content hash invariants."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_compression_and_metadata_invariance(self):
        """Content hash must be bit-identical despite different compression or metadata."""
        n = 500
        data = np.zeros((n, 5), dtype=np.float32)
        data[:, 0] = np.linspace(100.0, 50000.0, n)
        data[:, 1] = 4500.0
        data[:, 2] = 1.5
        data[:, 3] = 90.0
        data[:, 4] = -30.0
        labels = np.array([i % 4 for i in range(n)], dtype=np.int64)

        f_raw = _make_h5(self.root / "raw.h5", data, labels, compression=None)
        f_gz1 = _make_h5(self.root / "gz1.h5", data, labels, compression="gzip", compression_opts=1, attrs={"author": "alice"})
        f_gz9 = _make_h5(self.root / "gz9.h5", data, labels, compression="gzip", compression_opts=9, attrs={"author": "bob", "rev": 42})

        hash_raw = streaming_canonical_content_sha256(f_raw)
        hash_gz1 = streaming_canonical_content_sha256(f_gz1)
        hash_gz9 = streaming_canonical_content_sha256(f_gz9)

        # Content hashes must be identical
        self.assertEqual(hash_raw, hash_gz1)
        self.assertEqual(hash_raw, hash_gz9)

        # Raw file byte hashes must differ due to compression and attributes
        with open(f_raw, "rb") as f1, open(f_gz9, "rb") as f2:
            self.assertNotEqual(f1.read(), f2.read())

    def test_hash_sensitivity_to_data_and_label_modifications(self):
        """Content hash must change if any data value or emitter label is altered."""
        n = 200
        data = np.zeros((n, 5), dtype=np.float32)
        data[:, 0] = np.arange(n, dtype=np.float32) * 100.0
        data[:, 1] = 3000.0
        data[:, 2] = 2.0
        data[:, 3] = 45.0
        data[:, 4] = -20.0
        labels = np.zeros(n, dtype=np.int64)

        base_file = _make_h5(self.root / "base.h5", data, labels)
        base_hash = streaming_canonical_content_sha256(base_file)

        # Modify one frequency value
        data_mod = data.copy()
        data_mod[5, 1] = 3001.0
        mod_f_file = _make_h5(self.root / "mod_f.h5", data_mod, labels)
        self.assertNotEqual(base_hash, streaming_canonical_content_sha256(mod_f_file))

        # Modify one label value
        labels_mod = labels.copy()
        labels_mod[10] = 1
        mod_lbl_file = _make_h5(self.root / "mod_lbl.h5", data, labels_mod)
        self.assertNotEqual(base_hash, streaming_canonical_content_sha256(mod_lbl_file))


class Phase10SplitIsolationTests(unittest.TestCase):
    """Verify 3-layer cross-split isolation."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_clean_disjoint_splits_pass(self):
        train_file = _make_h5(self.root / "train" / "p1.h5", np.ones((50, 5)), np.zeros(50))
        val_file = _make_h5(self.root / "val" / "p2.h5", np.full((50, 5), 2.0), np.ones(50))
        test_file = _make_h5(self.root / "test" / "p3.h5", np.full((50, 5), 3.0), np.full(50, 2))

        report = validate_split_isolation(
            {"train": [train_file], "val": [val_file], "test": [test_file]},
            root=self.root,
            fail_fast=True,
        )
        self.assertTrue(report["isolated"])
        self.assertEqual(len(report["layer1_path_overlaps"]), 0)
        self.assertEqual(len(report["layer2_raw_hash_overlaps"]), 0)
        self.assertEqual(len(report["layer3_content_hash_overlaps"]), 0)

    def test_layer1_path_overlap_fails_closed(self):
        train_file = _make_h5(self.root / "shared" / "same.h5", np.ones((50, 5)), np.zeros(50))
        # Pointing to the same file path in both train and val
        with self.assertRaises(SplitLeakageError):
            validate_split_isolation(
                {"train": [train_file], "val": [train_file]},
                root=self.root,
                fail_fast=True,
            )

    def test_layer2_raw_hash_overlap_fails_closed(self):
        # Different paths, identical bytes
        f1 = _make_h5(self.root / "train" / "f1.h5", np.ones((50, 5)), np.zeros(50))
        f2 = self.root / "val" / "f2.h5"
        f2.parent.mkdir(parents=True, exist_ok=True)
        f2.write_bytes(f1.read_bytes())

        with self.assertRaises(SplitLeakageError):
            validate_split_isolation(
                {"train": [f1], "val": [f2]},
                root=self.root,
                fail_fast=True,
            )

    def test_layer3_canonical_content_overlap_fails_closed(self):
        # Different compression / headers / file bytes, but identical pulse trains
        data = np.ones((50, 5), dtype=np.float32)
        labels = np.zeros(50, dtype=np.int64)

        f_train = _make_h5(self.root / "train" / "f_train.h5", data, labels, compression="gzip", compression_opts=1)
        f_test = _make_h5(self.root / "test" / "f_test.h5", data, labels, compression="gzip", compression_opts=9, attrs={"variant": "v2"})

        # Raw file bytes differ
        self.assertNotEqual(f_train.read_bytes(), f_test.read_bytes())

        # Layer 3 detects exact semantic content duplication across train and test!
        with self.assertRaises(SplitLeakageError):
            validate_split_isolation(
                {"train": [f_train], "test": [f_test]},
                root=self.root,
                fail_fast=True,
            )


class Phase10TaxonomyAndProvenanceTests(unittest.TestCase):
    """Verify taxonomy classification and provenance role contracts."""

    def test_taxonomy_classes_complete(self):
        self.assertEqual(len(PROJECT_TAXONOMY_CLASSES), 8)
        self.assertIn("fixed", PROJECT_TAXONOMY_CLASSES)
        self.assertIn("fast_agile", PROJECT_TAXONOMY_CLASSES)
        self.assertIn("slow_agile", PROJECT_TAXONOMY_CLASSES)
        self.assertIn("markov_agile", PROJECT_TAXONOMY_CLASSES)
        self.assertIn("periodic_scan", PROJECT_TAXONOMY_CLASSES)
        self.assertIn("mixed", PROJECT_TAXONOMY_CLASSES)
        self.assertIn("dense", PROJECT_TAXONOMY_CLASSES)
        self.assertIn("sparse", PROJECT_TAXONOMY_CLASSES)

    def test_unknown_fallback_on_empty_scenario(self):
        res = classify_project_taxonomy(np.zeros((0, 5)), np.zeros(0))
        self.assertEqual(res["primary_class"], "unknown")
        self.assertEqual(res["classification_confidence"], 0.0)
        self.assertIn("empty", res["classification_evidence"])

    def test_near_duplicate_diagnostic_runs(self):
        data = np.linspace(0.0, 1000.0, 500).reshape(100, 5).astype(np.float32)
        diag = compute_near_duplicate_diagnostic(data)
        self.assertIsInstance(diag, str)
        self.assertEqual(len(diag), 16)
        diag2 = compute_near_duplicate_diagnostic(data)
        self.assertEqual(diag, diag2)

    def test_role_contract_rejects_train_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError) as ctx:
                build_manifest(root, evaluation_split="train")
            self.assertIn("Role contract violation", str(ctx.exception))

    def test_strict_non_decreasing_toa_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            p_valid = Path(tmp) / "valid.h5"
            p_invalid = Path(tmp) / "invalid.h5"

            data_valid = np.zeros((10, 5), dtype=np.float32)
            data_valid[:, 0] = np.arange(10, dtype=np.float32) * 100.0
            data_valid[:, 1] = 5000.0
            data_valid[:, 2] = 1.0
            _make_h5(p_valid, data_valid, np.zeros(10))

            data_invalid = data_valid.copy()
            data_invalid[5, 0] = 50.0  # ToA drops backwards!
            _make_h5(p_invalid, data_invalid, np.zeros(10))

            validator = TSRDValidator()
            v_val = validator.validate_file(p_valid)
            self.assertTrue(v_val["valid"])

            v_inval = validator.validate_file(p_invalid)
            self.assertFalse(v_inval["valid"])
            self.assertIn("ToA is not monotonically non-decreasing", v_inval["errors"])


class Phase10ControlledBatteryTests(unittest.TestCase):
    """Gate 10.4B: Verify controlled synthetic fixture battery coverage."""

    def test_fixture_battery_has_minimum_3_per_class(self):
        fixture_root = Path("tests/fixtures/phase10_tsrd")
        self.assertTrue(fixture_root.exists(), f"Fixture root {fixture_root} must exist")

        for cls_name in PROJECT_TAXONOMY_CLASSES:
            cls_dir = fixture_root / cls_name
            self.assertTrue(cls_dir.exists(), f"Directory for class {cls_name} missing")
            h5_files = list(cls_dir.glob("*.h5"))
            self.assertGreaterEqual(
                len(h5_files),
                3,
                f"Class {cls_name} has only {len(h5_files)} scenarios (minimum 3 required for Gate 10.4B)",
            )
            for f in h5_files:
                with h5py.File(str(f), "r") as handle:
                    data = np.asarray(handle["data"])
                    labels = np.asarray(handle["labels"]).reshape(-1)
                    res = classify_project_taxonomy(data, labels)
                    self.assertEqual(
                        res["primary_class"],
                        cls_name,
                        f"Fixture {f} was expected to be {cls_name}, got {res['primary_class']}",
                    )


class Phase10RemediationRigorousTests(unittest.TestCase):
    """Rigorous scientific contract tests for Phase 10 remediation."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_physical_diagnostics_vs_structural_finiteness(self):
        """Zero PW and high amplitudes are physical diagnostics, not structural failures. NaNs/Infs are structural failures."""
        p_valid_diag = self.root / "valid_diag.h5"
        data_diag = np.zeros((10, 5), dtype=np.float32)
        data_diag[:, 0] = np.arange(10, dtype=np.float32) * 100.0
        data_diag[:, 1] = 9500.0
        data_diag[:, 2] = 0.0  # Zero PW (observed in TSRD test)
        data_diag[:, 3] = 30.0
        data_diag[:, 4] = 45.0  # Positive amplitude +45 dB (observed in TSRD train)
        _make_h5(p_valid_diag, data_diag, np.zeros(10))

        validator = TSRDValidator()
        v = validator.validate_file_streaming(p_valid_diag)
        self.assertTrue(v["structurally_valid"])
        self.assertEqual(v["nonfinite_count"], 0)
        self.assertTrue(v["observed_ranges"]["has_zero_pw"])
        self.assertEqual(v["observed_ranges"]["amplitude_range"], [45.0, 45.0])

        # Test NaN and Inf fail structurally
        p_nan = self.root / "nan.h5"
        data_nan = data_diag.copy()
        data_nan[2, 1] = np.nan
        data_nan[4, 4] = np.inf
        _make_h5(p_nan, data_nan, np.zeros(10))

        v_nan = validator.validate_file_streaming(p_nan)
        self.assertFalse(v_nan["structurally_valid"])
        self.assertEqual(v_nan["nonfinite_count"], 2)

    def test_immutability_guard_tracks_and_detects_alterations(self):
        """DatasetImmutabilityGuard must detect byte modifications, size changes, and deletions."""
        from ew_core.data.tsrd_manifest import DatasetImmutabilityGuard

        f1 = self.root / "file1.h5"
        f2 = self.root / "file2.h5"
        _make_h5(f1, np.ones((10, 5)), np.zeros(10))
        _make_h5(f2, np.full((10, 5), 2.0), np.ones(10))

        guard = DatasetImmutabilityGuard()
        guard.record(f1)
        guard.record(f2)

        status = guard.verify_all()
        self.assertTrue(status["passed"])
        self.assertEqual(len(status["violations"]), 0)

        # Append byte to f1
        with open(f1, "ab") as handle:
            handle.write(b"corrupt")

        status_mut = guard.verify_all()
        self.assertFalse(status_mut["passed"])
        self.assertEqual(len(status_mut["violations"]), 1)
        self.assertEqual(status_mut["violations"][0]["type"], "size_mismatch")

        # Delete f2
        f2.unlink()
        status_del = guard.verify_all()
        self.assertFalse(status_del["passed"])
        types = [v["type"] for v in status_del["violations"]]
        self.assertIn("missing_file", types)

    def test_transmitter_metadata_consistency_audit(self):
        """Transmitter metadata audit correctly flags consistent vs inconsistent entries."""
        from ew_core.data.tsrd_manifest import audit_tsrd_transmitter_metadata

        p_stare = self.root / "stare_sample.h5"
        data = np.zeros((20, 5), dtype=np.float32)
        data[:, 0] = np.arange(20, dtype=np.float32) * 50.0
        data[:, 1] = 3000.0  # 3000 MHz
        data[:, 2] = 2.0     # 2.0 us
        labels = np.zeros(20, dtype=np.int64)

        with h5py.File(str(p_stare), "w") as h:
            h.create_dataset("data", data=data)
            h.create_dataset("labels", data=labels.reshape(-1, 1))
            meta = h.create_group("metadata")
            tx = meta.create_group("transmitters")
            tx0 = tx.create_group("transmitters_0")
            fc = tx0.create_group("frequency_config")
            fc.create_dataset("freqs_mhz", data=np.array([3000.0]))
            pwc = tx0.create_group("pulse_width_config")
            pwc.create_dataset("pws_us", data=np.array([2.0]))

        audit_res = audit_tsrd_transmitter_metadata(p_stare)
        self.assertEqual(audit_res["tier"], "consistent")
        self.assertTrue(audit_res["consistent"])
        self.assertFalse(audit_res["quarantined_for_metadata_dependent_training"])

    def test_path_bound_manifest_fingerprint_sensitivity(self):
        """Global manifest fingerprint must be sensitive to path/mode/split changes."""
        from ew_core.data.tsrd_manifest import build_integrated_manifest

        d_root = self.root / "tsrd_mock"
        for mode in ["scan", "stare"]:
            for split in ["train", "val", "test"]:
                sp = d_root / mode / f"{split}_{mode}"
                sp.mkdir(parents=True, exist_ok=True)
                _make_h5(sp / "config_0.h5", np.ones((10, 5)), np.zeros(10))

        m1 = build_integrated_manifest(d_root, output_path=self.root / "m1.json", enforce_split_isolation=False)
        fp1 = m1["dataset_fingerprint"]

        # Rename one file
        src = d_root / "scan" / "test_scan" / "config_0.h5"
        dst = d_root / "scan" / "test_scan" / "config_1.h5"
        src.rename(dst)

        m2 = build_integrated_manifest(d_root, output_path=self.root / "m2.json", enforce_split_isolation=False)
        fp2 = m2["dataset_fingerprint"]

        self.assertNotEqual(fp1, fp2)


if __name__ == "__main__":
    unittest.main()

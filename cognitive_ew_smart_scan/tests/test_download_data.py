import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

from scripts import download_data as dd


def _make_h5(path: Path, rows: int = 100, cols: int = 5, toa0: float = 1000.0, toa_step: float = 10.0,
             labels: list[int] | None = None, data=None, labels_as_column: bool = False,
             collection_time_s: float | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w") as f:
        if data is None:
            toa = np.arange(rows, dtype=np.float64) * toa_step + toa0
            data = np.column_stack([toa, np.full(rows, 10e9), np.full(rows, 1e-6), np.zeros(rows), np.full(rows, 0.5)])
        f.create_dataset("data", data=np.asarray(data))
        if labels is None:
            labels = [i % 3 for i in range(rows)]
        lab = np.asarray(labels, dtype=np.int64)
        if labels_as_column:
            lab = lab.reshape(-1, 1)
        f.create_dataset("labels", data=lab)
        if collection_time_s is not None:
            f.create_dataset("metadata", data=h5py.Empty("f8"))
            f["metadata"].attrs["collection_time_s"] = np.float64(collection_time_s)
    return path


class DownerModesAndSplitsTests(unittest.TestCase):
    def test_defaults_normalise(self):
        self.assertEqual(dd._normalise_modes(None), ["stare", "scan"])
        self.assertEqual(dd._normalise_splits(None), ["train", "validation", "test"])

    def test_val_alias(self):
        self.assertEqual(dd._normalise_splits(["val"]), ["validation"])
        self.assertEqual(dd._normalise_splits(["valid"]), ["validation"])

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            dd._normalise_modes(["foobar"])

    def test_invalid_split_raises(self):
        with self.assertRaises(ValueError):
            dd._normalise_splits(["juice"])


class BelongsToTests(unittest.TestCase):
    def test_directory_path(self):
        self.assertTrue(dd._belongs_to("stare/train/a.h5", "stare", "train"))
        self.assertTrue(dd._belongs_to("stare/train/x/y/a.h5", "stare", "train"))
        self.assertFalse(dd._belongs_to("stare/validation/a.h5", "stare", "train"))

    def test_filename_fallback(self):
        self.assertTrue(dd._belongs_to("data_stare_train_001.h5", "stare", "train"))
        self.assertFalse(dd._belongs_to("data_stare_test_001.h5", "stare", "train"))

    def test_official_kaggle_scan_dir_names(self):
        # Official TSRD store dirs: <split>_<mode>/ (requirement 2).
        self.assertTrue(dd._belongs_to("scan/train_scan/config_0.h5", "scan", "train"))
        self.assertTrue(dd._belongs_to("scan/val_scan/config_7.h5", "scan", "validation"))
        self.assertTrue(dd._belongs_to("scan/test_scan/config_3.h5", "scan", "test"))

    def test_official_kaggle_stare_dir_names(self):
        self.assertTrue(dd._belongs_to("stare/train_stare/config_0.h5", "stare", "train"))
        self.assertTrue(dd._belongs_to("stare/val_stare/config_9.h5", "stare", "validation"))
        self.assertTrue(dd._belongs_to("stare/test_stare/config_2.h5", "stare", "test"))

    def test_kaggle_names_do_not_cross_modes(self):
        self.assertFalse(dd._belongs_to("scan/train_scan/a.h5", "stare", "train"))
        self.assertFalse(dd._belongs_to("stare/val_stare/a.h5", "scan", "validation"))
        self.assertFalse(dd._belongs_to("scan/test_scan/a.h5", "scan", "train"))

    def test_each_split_gets_its_own_kaggle_dir(self):
        # train selection must not swallow val/test_dirs.
        f = "scan/train_scan/config_0.h5"
        self.assertTrue(dd._belongs_to(f, "scan", "train"))
        self.assertFalse(dd._belongs_to(f, "scan", "validation"))
        self.assertFalse(dd._belongs_to(f, "scan", "test"))
        f = "scan/val_scan/config_0.h5"
        self.assertTrue(dd._belongs_to(f, "scan", "validation"))
        self.assertFalse(dd._belongs_to(f, "scan", "train"))


class Sha256Tests(unittest.TestCase):
    def test_sha256_matches_hashlib(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.h5"
            _make_h5(p, rows=5)
            blob = p.read_bytes()
            self.assertEqual(dd._sha256(p), hashlib.sha256(blob).hexdigest())


class VerifyH5Tests(unittest.TestCase):
    def test_happy_path_all_ten_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _make_h5(Path(tmp) / "ok.h5", rows=120, toa0=500.0, toa_step=2.5,
                         labels=[i % 3 for i in range(120)])
            rec = dd._verify_h5(p)
            self.assertEqual(rec.pulse_count, 120)
            self.assertEqual(rec.emitter_count, 3)
            self.assertEqual(rec.emitter_count_incl_noise, 3)
            self.assertAlmostEqual(rec.duration_us, 120 * 2.5 - 2.5)
            self.assertAlmostEqual(rec.toa_min_us, 500.0)
            self.assertEqual(rec.shape, [120, 5])
            self.assertEqual(len(rec.sha256), 64)

    def test_official_labels_column_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _make_h5(Path(tmp) / "col.h5", rows=40, labels_as_column=True, collection_time_s=30.0)
            rec = dd._verify_h5(p)
            self.assertEqual(rec.pulse_count, 40)
            self.assertEqual(rec.emitter_count, 3)
            self.assertEqual(rec.collection_time_s, 30.0)

    def test_noise_label_excluded_from_emitter_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _make_h5(Path(tmp) / "noisy.h5", rows=50, labels=([-1] * 25) + ([i % 3 for i in range(25)]))
            rec = dd._verify_h5(p)
            self.assertEqual(rec.emitter_count, 3)
            self.assertEqual(rec.emitter_count_incl_noise, 4)

    def test_missing_data_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "nodata.h5"
            with h5py.File(str(p), "w") as f:
                f.create_dataset("labels", data=np.zeros(10))
            with self.assertRaises(ValueError):
                dd._verify_h5(p)

    def test_missing_labels_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "nolabels.h5"
            with h5py.File(str(p), "w") as f:
                f.create_dataset("data", data=np.zeros((10, 5)))
            with self.assertRaises(ValueError):
                dd._verify_h5(p)

    def test_wrong_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "sixcol.h5"
            with h5py.File(str(p), "w") as f:
                f.create_dataset("data", data=np.zeros((10, 6)))
                f.create_dataset("labels", data=np.zeros(10))
            with self.assertRaises(ValueError):
                dd._verify_h5(p)

    def test_label_length_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "shortlabels.h5"
            with h5py.File(str(p), "w") as f:
                f.create_dataset("data", data=np.zeros((10, 5)))
                f.create_dataset("labels", data=np.zeros(7))
            with self.assertRaises(ValueError):
                dd._verify_h5(p)

    def test_unreadable_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "not_h5.h5"
            p.write_bytes(b"this is not an hdf5 file")
            with self.assertRaises(ValueError):
                dd._verify_h5(p)

    def test_non_finite_values_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = np.zeros((10, 5))
            data[3, 0] = np.nan
            p = _make_h5(Path(tmp) / "nan.h5", data=data, labels=[0] * 10)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                dd._verify_h5(p)

    def test_unordered_toa_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            toa = np.array([0, 10, 5, 20], dtype=np.float64)
            data = np.column_stack([toa, np.full(4, 9e9), np.full(4, 1e-6), np.zeros(4), np.full(4, 0.5)])
            p = _make_h5(Path(tmp) / "unsorted.h5", data=data, labels=[0, 1, 0, 1])
            with self.assertRaisesRegex(ValueError, "non-decreasing"):
                dd._verify_h5(p)

    def test_zero_pulse_official_scene_is_valid(self):
        # Official TSRD contains legitimate (0,5)/(0,1) empty scenes.
        with tempfile.TemporaryDirectory() as tmp:
            p = _make_h5(Path(tmp) / "empty.h5", rows=0, cols=5)
            rec = dd._verify_h5(p)
            self.assertEqual(rec.pulse_count, 0)
            self.assertEqual(rec.duration_us, 0.0)
            self.assertEqual(len(rec.sha256), 64)


class FullAcquisitionTests(unittest.TestCase):
    """Offline end-to-end: fake huggingface_hub so nothing touches the network."""

    def _run(self, output_dir: Path, repo_files, downloaded, *,
             modes=None, splits=None, allow=True, dry_run=False):
        def fake_list_repo_files(repo_id, token=None):
            return list(repo_files)

        def fake_hf_hub_download(repo_id, filename, local_dir, token=None):
            if filename not in downloaded:
                raise RuntimeError(f"synthetic network failure for {filename}")
            local = Path(local_dir) / filename
            for i, row in enumerate(downloaded[filename]):
                toa = np.arange(20, dtype=np.float64) * 1.0 + 1000.0 * i
                data = np.column_stack([toa, np.full(20, 9e9), np.full(20, 1e-6), np.zeros(20), np.full(20, 0.25)])
                labels = [n % 2 for n in range(20)]
                _make_h5(local, data=data, labels=labels)
            return str(local)

        with mock.patch.object(
            dd, "_hub_functions", return_value=(fake_hf_hub_download, fake_list_repo_files)
        ):
            return dd.download_tsr_dataset(
                output_dir=output_dir, token="hf_placeholder", modes=modes, splits=splits,
                allow_download=allow, dry_run=dry_run,
            )

    def test_skipped_without_allow_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = self._run(Path(tmp), ["stare/train/1.h5"], {}, allow=False, modes=["stare"], splits=["train"])
            self.assertEqual(summary["status"], "skipped")
            self.assertFalse((Path(tmp) / "stare").exists())

    def test_canonic_acquire_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            summary = self._run(
                out,
                [
                    "stare/train/a.h5",
                    "scan/validation/b.h5",
                    "stare/validation/c.h5",
                    "scan/train/d.h5",
                ],
                {
                    "stare/train/a.h5": ["row0"],
                    "scan/validation/b.h5": ["row0"],
                    "stare/validation/c.h5": ["row0"],
                    "scan/train/d.h5": ["row0"],
                },
                modes=["stare", "scan"], splits=["train", "validation"],
            )
            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["verified_files"], 4)
            self.assertEqual(summary["failed_files"], 0)
            self.assertEqual(summary["missing"], [])
            self.assertTrue((out / "stare" / "train" / "a.h5").exists())
            self.assertTrue((out / "scan" / "validation" / "b.h5").exists())
            sub = json.loads((out / "stare" / "train" / "manifest.json").read_text())
            self.assertEqual(sub["status"], "ok")
            self.assertEqual(sub["files"][0]["pulse_count"], 20)
            self.assertEqual(sub["files"][0]["emitter_count"], 2)
            agg = json.loads((out / "manifest.json").read_text())
            self.assertEqual(agg["totals"]["pulses"], 80)
            self.assertEqual(agg["subsets"]["stare/train"]["status"], "ok")

    def test_missing_subset_recorded_never_fabricated(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            summary = self._run(
                out,
                ["stare/train/a.h5"],
                {"stare/train/a.h5": ["row0"]},
                modes=["stare", "scan"], splits=["train"],
            )
            self.assertEqual(summary["status"], "partial")
            self.assertIn("scan/train", summary["missing"])
            self.assertEqual(list((out / "scan" / "train").glob("*.h5")), [])
            sub = json.loads((out / "scan" / "train" / "manifest.json").read_text())
            self.assertEqual(sub["status"], "missing")
            self.assertEqual(sub["files"], [])

    def test_failed_download_recorded_without_silent_finalise(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            summary = self._run(
                out,
                ["stare/train/a.h5", "stare/train/bad.h5"],
                {"stare/train/a.h5": ["row0"]},
                modes=["stare"], splits=["train"],
            )
            self.assertEqual(summary["status"], "partial")
            self.assertEqual(summary["verified_files"], 1)
            self.assertEqual(summary["failed_files"], 1)
            sub = json.loads((out / "stare" / "train" / "manifest.json").read_text())
            self.assertEqual([f["name"] for f in sub["failed_files"]], ["stare/train/bad.h5"])
            self.assertIn("reason", sub["failed_files"][0])
            self.assertFalse((out / "stare" / "train" / "bad.h5").exists())

    def test_dry_run_downloads_nothing_writes_no_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            summary = self._run(
                out,
                ["stare/train/a.h5", "stare/train/b.h5", "scan/train/c.h5"],
                {"stare/train/a.h5": ["row0"], "stare/train/b.h5": ["row0"], "scan/train/c.h5": ["row0"]},
                modes=["stare", "scan"], splits=["train"],
                dry_run=True,
            )
            self.assertEqual(summary["status"], "dry_run")
            self.assertEqual(summary["would_download_files"], 3)
            self.assertEqual(summary["subsets"]["stare/train"]["planned_files"], 2)
            # No directories, no files, no manifests created.
            self.assertEqual(list(out.rglob("*.h5")), [])
            self.assertEqual(list(out.rglob("manifest.json")), [])
            self.assertFalse((out / "stare").exists())

    def test_kaggle_source_lands_in_canonical_tree(self):
        # Official Kaggle repo paths (<split>_<mode>) must land at the canonical
        # <mode>/<split>/ layout (requirement 1), byte-for-byte.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            summary = self._run(
                out,
                ["scan/train_scan/config_0.h5", "scan/val_scan/config_7.h5", "scan/test_scan/config_3.h5"],
                {
                    "scan/train_scan/config_0.h5": ["row0"],
                    "scan/val_scan/config_7.h5": ["row0"],
                    "scan/test_scan/config_3.h5": ["row0"],
                },
                modes=["scan"], splits=["train", "validation", "test"],
            )
            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["verified_files"], 3)
            self.assertFalse((out / "scan" / "train_scan").exists())
            self.assertTrue((out / "scan" / "train" / "config_0.h5").exists())
            self.assertTrue((out / "scan" / "validation" / "config_7.h5").exists())
            self.assertTrue((out / "scan" / "test" / "config_3.h5").exists())
            agg = json.loads((out / "manifest.json").read_text())
            # Requirement 7: aggregate file count reflects verified files.
            self.assertEqual(agg["totals"]["files"], 3)


class DeprecatedDownloadShimTests(unittest.TestCase):
    """The legacy download_tsrd.py must forward to the authoritative path."""

    def test_shim_forwards_to_authoritative_without_allow_download(self):
        from scripts import download_tsrd

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(download_tsrd, "download_tsr_dataset") as fake:
                fake.return_value = {
                    "status": "skipped",
                    "reason": "download_disabled_by_default",
                    "output_dir": tmp,
                }
                with mock.patch.object(
                    download_tsrd.sys, "argv", ["download_tsrd.py", "--output-dir", tmp, "--modes", "scan", "--splits", "train"]
                ):
                    with mock.patch.object(download_tsrd.sys, "stderr", new=mock.MagicMock()):
                        download_tsrd.main()
                fake.assert_called_once()
                kwargs = fake.call_args.kwargs
                self.assertEqual(kwargs["output_dir"], tmp)
                self.assertEqual(kwargs["modes"], ["scan"])
                self.assertEqual(kwargs["splits"], ["train"])
                self.assertFalse(kwargs["allow_download"])

    def test_shim_follows_env_root_default(self):
        from scripts import download_tsrd

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"TSRD_DATA_ROOT": tmp}):
            with mock.patch.object(download_tsrd, "download_tsr_dataset") as fake:
                fake.return_value = {"status": "skipped"}
                with mock.patch.object(
                    download_tsrd.sys, "argv", ["download_tsrd.py"]
                ):
                    with mock.patch.object(download_tsrd.sys, "stderr", new=mock.MagicMock()):
                        download_tsrd.main()
                self.assertEqual(fake.call_args.kwargs["output_dir"], tmp)

    def test_shim_forwards_dry_run_flag(self):
        from scripts import download_tsrd

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(download_tsrd, "download_tsr_dataset") as fake:
                fake.return_value = {"status": "dry_run", "would_download_files": 2}
                with mock.patch.object(
                    download_tsrd.sys, "argv", ["download_tsrd.py", "--dry-run"]
                ):
                    with mock.patch.object(download_tsrd.sys, "stderr", new=mock.MagicMock()):
                        download_tsrd.main()
                self.assertTrue(fake.call_args.kwargs["dry_run"])


if __name__ == "__main__":
    unittest.main()
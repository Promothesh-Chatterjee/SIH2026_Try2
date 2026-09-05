"""Unit tests for the canonical TSRD root/split-layout contract.

Covers ``src/data/tsrd_root.py``: root-resolution precedence, pathlib
normalization, the Kaggle/conventional/archive/flat split-layout aliases, and
the real-TSRD no-synthetic-substitution guard.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.data.tsrd_manifest import resolve_split_dirs
from src.data.tsrd_root import (
    DEFAULT_DATA_DIR,
    resolve_config_data_dir,
    resolve_split_dir,
    resolve_tsrd_root,
)


class ResolverPrecedenceTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def _write_config(self, data_dir: str) -> Path:
        cfg = Path(self.root) / "training_config.yaml"
        cfg.write_text(f"data_dir: {data_dir!r}\n", encoding="utf-8")
        return cfg

    def test_default_is_relative_repo_safe(self):
        self.assertEqual(resolve_tsrd_root(), Path(DEFAULT_DATA_DIR))

    def test_config_data_dir_used(self):
        cfg = self._write_config("D:/TSRD")
        self.assertEqual(resolve_tsrd_root(config={"data_dir": "D:/TSRD"}), Path("D:/TSRD"))

    def test_env_overrides_config(self):
        cfg = self._write_config("D:/ignored")
        with mock.patch.dict(os.environ, {"TSRD_DATA_ROOT": "D:/TSRD"}):
            self.assertEqual(resolve_tsrd_root(config={"data_dir": "D:/ignored"}), Path("D:/TSRD"))

    def test_cli_overrides_env(self):
        with mock.patch.dict(os.environ, {"TSRD_DATA_ROOT": "D:/env"}):
            self.assertEqual(resolve_tsrd_root("D:/cli"), Path("D:/cli"))

    def test_cli_overrides_config(self):
        cfg = self._write_config("D:/ignored")
        self.assertEqual(resolve_tsrd_root("D:/cli", config={"data_dir": "D:/ignored"}), Path("D:/cli"))

    def test_resolve_config_data_dir_reads_yaml(self):
        cfg = self._write_config("D:/TSRD")
        self.assertEqual(resolve_config_data_dir(cfg), Path("D:/TSRD"))
        self.assertEqual(resolve_config_data_dir(cfg, cli_value="D:/cli"), Path("D:/cli"))

    def test_env_clear_returns_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_tsrd_root(), Path("data"))


class PathNormalisationTests(unittest.TestCase):
    def test_win_and_posix_separators_are_equivalent(self):
        self.assertEqual(Path("D:/TSRD"), Path("D:" + os.sep + "TSRD") if os.sep == "\\" else Path("D:\\TSRD"))
        win = resolve_tsrd_root("D:\\TSRD\\scan")
        posix = resolve_tsrd_root("D:/TSRD/scan")
        self.assertIsInstance(win, Path)
        self.assertEqual(win.parts, posix.parts)

    def test_resolver_returns_path_objects(self):
        with mock.patch.dict(os.environ, {"TSRD_DATA_ROOT": "D:/TSRD"}):
            self.assertIsInstance(resolve_tsrd_root(), Path)


class _LayoutFixtures(unittest.TestCase):
    def _seed(self, data_root: Path, dirs: dict[str, str]):
        """Create the given dirs relative to data_root (POSIX-style)."""
        for rel in dirs.values():
            (data_root / rel).mkdir(parents=True, exist_ok=True)

    def _assert_scan(self, data_root: Path, train: str, val: str, test: str):
        resolved = resolve_split_dirs(data_root, mode="scan")
        self.assertEqual(resolved.keys(), {"train", "val", "test"})
        self.assertEqual(Path(resolved["train"]).resolve(), (data_root / train).resolve())
        self.assertEqual(Path(resolved["val"]).resolve(), (data_root / val).resolve())
        self.assertEqual(Path(resolved["test"]).resolve(), (data_root / test).resolve())


class KaggleLayoutTests(_LayoutFixtures):
    def test_kaggle_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._seed(
                root,
                {
                    "scan/train_scan": "scan/train_scan",
                    "scan/val_scan": "scan/val_scan",
                    "scan/test_scan": "scan/test_scan",
                },
            )
            self._assert_scan(root, "scan/train_scan", "scan/val_scan", "scan/test_scan")

    def test_kaggle_aliases_stare(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._seed(
                root,
                {
                    "stare/train_stare": "stare/train_stare",
                    "stare/val_stare": "stare/val_stare",
                    "stare/test_stare": "stare/test_stare",
                },
            )
            resolved = resolve_split_dirs(root, mode="stare")
            self.assertEqual(Path(resolved["train"]).resolve(), (root / "stare/train_stare").resolve())
            self.assertEqual(Path(resolved["val"]).resolve(), (root / "stare/val_stare").resolve())
            self.assertEqual(Path(resolved["test"]).resolve(), (root / "stare/test_stare").resolve())


class CanonicalStructureTests(_LayoutFixtures):
    """Phase 1 requirement 4: the canonical dataset structure.

    ``<root>/<mode>/{train,validation,test}`` for both ``scan`` and ``stare``,
    with the full-word ``validation`` alias resolved transparently.
    """

    def test_canonical_structure_scan_and_stare(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._seed(
                root,
                {
                    "scan/train": "scan/train",
                    "scan/validation": "scan/validation",
                    "scan/test": "scan/test",
                    "stare/train": "stare/train",
                    "stare/validation": "stare/validation",
                    "stare/test": "stare/test",
                },
            )
            for mode in ("scan", "stare"):
                resolved = resolve_split_dirs(root, mode)
                self.assertEqual(
                    Path(resolved["train"]).resolve(), (root / mode / "train").resolve()
                )
                self.assertEqual(
                    Path(resolved["val"]).resolve(),
                    (root / mode / "validation").resolve(),
                )
                self.assertEqual(
                    Path(resolved["test"]).resolve(), (root / mode / "test").resolve()
                )

    def test_canonical_structure_via_resolve_split_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "stare" / "validation").mkdir(parents=True)
            self.assertEqual(
                Path(resolve_split_dir(root, "stare", "val")).resolve(),
                (root / "stare" / "validation").resolve(),
            )


class ConventionalLayoutTests(_LayoutFixtures):
    def test_conventional_train_val_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._seed(
                root,
                {"scan/train": "scan/train", "scan/val": "scan/val", "scan/test": "scan/test"},
            )
            self._assert_scan(root, "scan/train", "scan/val", "scan/test")

    def test_conventional_validation_full_word(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._seed(
                root,
                {
                    "scan/train": "scan/train",
                    "scan/validation": "scan/validation",
                    "scan/test": "scan/test",
                },
            )
            self._assert_scan(root, "scan/train", "scan/validation", "scan/test")

    def test_kaggle_alias_preferred_over_plain_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Both names exist: the Kaggle alias wins (first candidate).
            self._seed(
                root,
                {
                    "scan/train": "scan/train",
                    "scan/train_scan": "scan/train_scan",
                },
            )
            self.assertEqual(
                Path(resolve_split_dirs(root, "scan")["train"]).resolve(),
                (root / "scan/train_scan").resolve(),
            )


class ArchiveLayoutTests(_LayoutFixtures):
    def test_archive_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._seed(
                root,
                {
                    "scan/train_scan": "scan/train_scan",
                    "archive/train": "archive/train",
                    "archive/validation": "archive/validation",
                    "archive/test": "archive/test",
                },
            )
            # Kaggle train wins; archive supplies val/test when scan dirs missing.
            resolved = resolve_split_dirs(root, "scan")
            self.assertEqual(Path(resolved["train"]).resolve(), (root / "scan/train_scan").resolve())
            self.assertEqual(Path(resolved["val"]).resolve(), (root / "archive/validation").resolve())
            self.assertEqual(Path(resolved["test"]).resolve(), (root / "archive/test").resolve())


class FlatLayoutTests(_LayoutFixtures):
    def test_flat_kaggle_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._seed(root, {"train_scan": "train_scan", "val_scan": "val_scan", "test_scan": "test_scan"})
            resolved = resolve_split_dirs(root, "scan")
            self.assertEqual(Path(resolved["train"]).resolve(), (root / "train_scan").resolve())
            self.assertEqual(Path(resolved["val"]).resolve(), (root / "val_scan").resolve())
            self.assertEqual(Path(resolved["test"]).resolve(), (root / "test_scan").resolve())

    def test_mode_none_uses_scan_with_flat_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._seed(root, {"train_scan": "train_scan", "val_scan": "val_scan", "test_scan": "test_scan"})
            resolved = resolve_split_dirs(root, None)
            self.assertEqual(Path(resolved["train"]).resolve(), (root / "train_scan").resolve())

    def test_missing_split_falls_back_to_mode_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Only scan/train exists; test/val resolve to non-existing fallback
            # paths without raising.
            self._seed(root, {"scan/train": "scan/train"})
            resolved = resolve_split_dirs(root, "scan")
            self.assertEqual(Path(resolved["train"]).resolve(), (root / "scan/train").resolve())
            self.assertTrue(Path(resolved["val"]).suffix == "")  # a path, not None
            self.assertTrue(isinstance(resolved["test"], Path))


class ResolveSplitDirTests(unittest.TestCase):
    def test_invalid_split_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                resolve_split_dir(tmp, "scan", "bogus")

    def test_resolve_split_dir_single(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scan" / "train_scan").mkdir(parents=True)
            self.assertEqual(
                Path(resolve_split_dir(root, "scan", "train")).resolve(),
                (root / "scan/train_scan").resolve(),
            )


class RealTsrdNoSyntheticTests(unittest.TestCase):
    def test_scenario_source_never_silently_falls_back_in_real_tsrd(self):
        from src.environment.scenario_generator import ScenarioSource

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                ScenarioSource(
                    data_root=root,
                    mode="scan",
                    subset="train",
                    freq_min_mhz=0.0,
                    freq_max_mhz=18000.0,
                    seed=42,
                    allow_synthetic_fallback=False,
                )

    def test_scenario_source_synthetic_fallback_can_be_explicit(self):
        from src.environment.scenario_generator import ScenarioSource

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = ScenarioSource(
                data_root=root,
                mode="scan",
                subset="train",
                freq_min_mhz=0.0,
                freq_max_mhz=18000.0,
                seed=42,
                allow_synthetic_fallback=True,
            )
            self.assertEqual(source.source_label, "synthetic")
            records = source.sample()
            self.assertGreater(len(records), 0)
            self.assertTrue(all(getattr(r, "source_id", None) == "synthetic" for r in records))


if __name__ == "__main__":
    unittest.main()
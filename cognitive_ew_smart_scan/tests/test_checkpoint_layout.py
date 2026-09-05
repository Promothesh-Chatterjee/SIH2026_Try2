"""Phase 17: canonical checkpoint artifact layout.

All producers (trainers, export, eval scripts, API) must agree on one
structure and never write ambiguous root-level ``checkpoints/best.pt``-style
artifacts.
"""

import unittest
from pathlib import Path

import yaml

from src.utils.checkpoint_paths import (
    AMBIGUOUS_ARTIFACT_ROOTS,
    CHECKPOINT_ROOT,
    DEINTERLEAVER_ARTIFACTS,
    DEINTERLEAVER_DIR,
    ONNX_ARTIFACTS,
    ONNX_DIR,
    SCHEDULER_ARTIFACTS,
    SCHEDULER_DIR,
    resolve_checkpoint_dir,
)

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
CONFIG = ROOT / "configs" / "training_config.yaml"


class CanonicalLayoutTests(unittest.TestCase):
    def test_directories_under_checkpoints_root(self):
        for d in (DEINTERLEAVER_DIR, SCHEDULER_DIR, ONNX_DIR):
            self.assertTrue(str(d).startswith(str(CHECKPOINT_ROOT)))
            self.assertEqual(d.parent, CHECKPOINT_ROOT)

    def test_artifacts_match_contract(self):
        self.assertEqual(
            DEINTERLEAVER_ARTIFACTS,
            ("best.pt", "final.pt", "normalization_stats.json", "dataset_manifest.json", "metadata.json"),
        )
        self.assertEqual(SCHEDULER_ARTIFACTS, ("best.pt", "final.pt", "metadata.json"))
        self.assertEqual(ONNX_ARTIFACTS, ("deinterleaver.onnx", "scheduler.onnx"))

    def test_resolve_prefers_cli_override(self):
        self.assertEqual(
            resolve_checkpoint_dir(
                Path("experiments/custom"),
                "checkpoints",  # ignored — CLI wins
                SCHEDULER_DIR,
            ),
            Path("experiments/custom"),
        )

    def test_resolve_ambiguous_root_falls_back_to_canonical(self):
        for bad in ("checkpoints", "weights", "output", "mycheckpoints/checkpoints"):
            self.assertEqual(
                resolve_checkpoint_dir(None, bad, DEINTERLEAVER_DIR, role="deinterleaver"),
                DEINTERLEAVER_DIR,
            )

    def test_resolve_canonical_or_explicit_paths_preserved(self):
        self.assertEqual(resolve_checkpoint_dir(None, None, SCHEDULER_DIR), SCHEDULER_DIR)
        self.assertEqual(
            resolve_checkpoint_dir(None, "checkpoints/deinterleaver", DEINTERLEAVER_DIR),
            Path("checkpoints/deinterleaver"),
        )
        self.assertEqual(
            resolve_checkpoint_dir(None, "experiments/run1", SCHEDULER_DIR),
            Path("experiments/run1"),
        )

    def test_ambiguous_roots_include_checkpoints(self):
        self.assertIn("checkpoints", AMBIGUOUS_ARTIFACT_ROOTS)

    def test_trainers_use_resolver(self):
        from src.training import train_deinterleaver, train_scheduler

        src = (
            Path(train_deinterleaver.__file__).read_text(encoding="utf-8")
            + "\n"
            + Path(train_scheduler.__file__).read_text(encoding="utf-8")
        )
        self.assertGreaterEqual(src.count("resolve_checkpoint_dir("), 2)
        self.assertIn("DEINTERLEAVER_DIR", src)
        self.assertIn("SCHEDULER_DIR", src)

    def test_trainers_write_metadata_json(self):
        from src.training import train_deinterleaver, train_scheduler

        src = (
            Path(train_deinterleaver.__file__).read_text(encoding="utf-8")
            + "\n"
            + Path(train_scheduler.__file__).read_text(encoding="utf-8")
        )
        self.assertGreaterEqual(src.count("write_checkpoint_metadata("), 2)

    def test_export_onnx_writes_onnx_metadata(self):
        src = (ROOT / "src" / "deployment" / "export_onnx.py").read_text(encoding="utf-8")
        self.assertIn("write_checkpoint_metadata(", src)
        self.assertIn("out_dir / \"deinterleaver.onnx\"", src)
        self.assertIn("out_dir / \"scheduler.onnx\"", src)


class ScriptAgreementTests(unittest.TestCase):
    def _script(self, name):
        return (SCRIPTS / name).read_text(encoding="utf-8")

    def test_train_deinterleaver_script_canonical(self):
        self.assertIn("--output-dir checkpoints/deinterleaver/", self._script("train_deinterleaver.sh"))

    def test_train_scheduler_script_canonical(self):
        self.assertIn("--output-dir checkpoints/scheduler/", self._script("train_scheduler.sh"))

    def test_train_all_runs_both_canonical(self):
        s = self._script("train_all.sh")
        self.assertIn("train_deinterleaver.sh", s)
        self.assertIn("train_scheduler.sh", s)

    def test_evaluate_full_script_canonical(self):
        s = self._script("evaluate_full.sh")
        self.assertIn("checkpoints/deinterleaver/best.pt", s)
        self.assertIn("checkpoints/scheduler/best.pt", s)

    def test_no_script_writes_root_level_checkpoints(self):
        import re

        for name in ("train_deinterleaver.sh", "train_scheduler.sh", "evaluate_full.sh"):
            s = self._script(name)
            # Look for best.pt/final.pt under the raw checkpoints root only.
            for m in re.finditer(r"checkpoints/[\\/]?(best|final)\.pt", s):
                self.fail(f"{name} references root-level artifact {m.group(0)}")


class TrainingConfigTests(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    def test_no_ambiguous_root_output_dir(self):
        od = self.cfg.get("output_dir")
        if od is not None:
            self.assertNotIn(Path(str(od)).name, AMBIGUOUS_ARTIFACT_ROOTS)
            self.assertFalse(str(od).rstrip("/\\").endswith("checkpoints"))

    def test_checkpoint_references_are_canonical(self):
        self.assertEqual(
            self.cfg.get("deinterleaver_ckpt", "checkpoints/deinterleaver/best.pt"),
            "checkpoints/deinterleaver/best.pt",
        )
        self.assertEqual(
            self.cfg.get("normalization_stats", "checkpoints/deinterleaver/normalization_stats.json"),
            "checkpoints/deinterleaver/normalization_stats.json",
        )


if __name__ == "__main__":
    unittest.main()
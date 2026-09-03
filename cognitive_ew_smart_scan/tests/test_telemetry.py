import unittest
import json
import time
from pathlib import Path

from src.telemetry.run_manager import RunManager
from src.telemetry.publisher import TelemetryPublisher


class RunManagerTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("temp_runs_test")
        if self.root.exists():
            import shutil
            shutil.rmtree(self.root)

    def tearDown(self):
        import shutil
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_run_dir_metadata_and_git_revision(self):
        run = RunManager(root=self.root, config={"drqn_scheduler": {"n_bands": 36}}, extras={"split": "train"})
        self.assertTrue(run.dir.is_dir())
        meta = json.loads((run.dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["config"]["drqn_scheduler"]["n_bands"], 36)
        self.assertEqual(meta["split"], "train")
        self.assertTrue((run.dir / "checkpoints").is_dir())
        self.assertIn("run_id", meta)
        run.write_git_revision()
        if (run.dir / "git_revision.txt").exists():
            self.assertTrue((run.dir / "git_revision.txt").read_text().strip())

    def test_emit_appends_jsonl(self):
        run = RunManager(root=self.root)
        run.emit(step=1, pd=0.5, pfa=0.1, band_priorities=[0.1, 0.2])
        run.emit(step=2, pd=0.6)
        lines = (run.dir / "telemetry.jsonl").read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 2)
        first = json.loads(lines[0])
        self.assertEqual(first["step"], 1)
        self.assertAlmostEqual(first["pd"], 0.5)
        self.assertEqual(first["band_priorities"], [0.1, 0.2])

    def test_write_normalization(self):
        run = RunManager(root=self.root)
        run.write_normalization({"mean": [1.0], "std": [0.5]})
        stats = json.loads((run.dir / "normalization.json").read_text(encoding="utf-8"))
        self.assertEqual(stats["mean"], [1.0])


class TelemetryPublisherTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("temp_runs_test")
        if self.root.exists():
            import shutil
            shutil.rmtree(self.root)

    def tearDown(self):
        import shutil
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_no_fabrication_when_empty(self):
        pub = TelemetryPublisher()
        latest = pub.latest()
        self.assertFalse(latest["live"])
        self.assertEqual(latest["live_message"], "no live telemetry yet")
        # A real update turns live on.
        pub.update(step=0, band_priorities=[0.0] * 36)
        self.assertTrue(pub.latest()["live"])
        self.assertEqual(pub.latest()["n_updates"], 1)

    def test_persistence_via_run_manager(self):
        run = RunManager(root=self.root)
        pub = TelemetryPublisher(run=run)
        pub.update(step=5, pd=0.7, band_priorities=[0.0, 1.0])
        lines = (run.dir / "telemetry.jsonl").read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        rec = json.loads(lines[0])
        self.assertEqual(rec["step"], 5)
        self.assertEqual(rec["band_priorities"], [0.0, 1.0])

    def test_history_recent_first(self):
        pub = TelemetryPublisher(max_history=3)
        for i in range(5):
            pub.update(step=i)
        hist = pub.history()
        self.assertEqual(len(hist), 3)
        self.assertEqual(hist[0]["step"], 4)


if __name__ == "__main__":
    unittest.main()
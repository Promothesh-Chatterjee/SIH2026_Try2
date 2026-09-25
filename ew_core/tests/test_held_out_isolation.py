"""Tests for Phase G & H Held-Out Test Set Isolation and Cryptographic Verification."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.evaluate_held_out_test_set import verify_scenario_hashes


class TestHeldOutIsolation(unittest.TestCase):
    def test_verify_scenario_hashes_success(self):
        """Pre-flight check succeeds when files exist and match hashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            f1 = p / "config_test1.h5"
            f1.write_bytes(b"test data 1")
            import hashlib
            sha1 = hashlib.sha256(b"test data 1").hexdigest()

            manifest = {
                "held_out_test_scenarios": {
                    "config_test1": {
                        "file": "config_test1.h5",
                        "sha256": sha1,
                    }
                }
            }

            verified = verify_scenario_hashes(p, manifest)
            self.assertIn("config_test1", verified)
            self.assertEqual(verified["config_test1"], sha1)

    def test_verify_scenario_hashes_missing_file_fails_closed(self):
        """Fails closed when a declared file is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            manifest = {
                "held_out_test_scenarios": {
                    "config_missing": {
                        "file": "config_missing.h5",
                        "sha256": "abcdef",
                    }
                }
            }
            with self.assertRaises(FileNotFoundError):
                verify_scenario_hashes(p, manifest)

    def test_verify_scenario_hashes_corrupted_sha_fails_closed(self):
        """Fails closed when file content does not match declared SHA."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            f1 = p / "config_bad.h5"
            f1.write_bytes(b"corrupted content")

            manifest = {
                "held_out_test_scenarios": {
                    "config_bad": {
                        "file": "config_bad.h5",
                        "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                    }
                }
            }
            with self.assertRaises(ValueError):
                verify_scenario_hashes(p, manifest)


if __name__ == "__main__":
    unittest.main()

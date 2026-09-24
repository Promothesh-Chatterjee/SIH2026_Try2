"""Contract hardening tests for benchmark artifact and API endpoints.

Ensures that:
- reports/benchmark_results.json strictly satisfies the 4-scheduler canonical schema.
- All numeric fields are finite numbers and not string placeholders.
- TP, FN, FP, TN counters are mathematically consistent with Pd and Pfa.
- /api/benchmark fails closed (HTTP 500) if corrupt or malformed.
- /health and /ready expose benchmark and checkpoint metadata.
"""

import json
import math
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from ew_core.deployment.api import app, validate_benchmark_payload

BENCHMARK_PATH = Path("reports/benchmark_results.json")
EXPECTED_SCHEDULERS = {
    "SmartScan_DRQN_MoE",
    "Random",
    "RoundRobin",
    "HighestOccupancy",
}
REQUIRED_NUMERIC_FIELDS = [
    "pd",
    "pfa",
    "sensitivity_dbm",
    "avg_intercept_rate",
    "avg_reward",
    "tp",
    "fn",
    "fp",
    "tn",
]


class BenchmarkContractHardeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BENCHMARK_PATH.exists():
            raise FileNotFoundError(f"Benchmark file not found: {BENCHMARK_PATH}")
        with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
            cls.data = json.load(f)

    def test_benchmark_has_metadata_and_schedulers(self):
        self.assertIn("metadata", self.data)
        self.assertIn("schedulers", self.data)
        metadata = self.data["metadata"]
        self.assertEqual(metadata.get("schema_version"), "2026.1-CANONICAL")
        self.assertIn("checkpoint_sha256", metadata)
        self.assertIn("dataset_fingerprint", metadata)

    def test_benchmark_has_exactly_four_schedulers(self):
        schedulers = self.data.get("schedulers", {})
        self.assertEqual(
            len(schedulers),
            4,
            f"Expected exactly 4 schedulers, found {len(schedulers)}: {list(schedulers.keys())}",
        )

    def test_benchmark_scheduler_names_are_exact(self):
        schedulers = self.data.get("schedulers", {})
        self.assertEqual(set(schedulers.keys()), EXPECTED_SCHEDULERS)

    def test_benchmark_required_numeric_fields_are_numeric(self):
        schedulers = self.data.get("schedulers", {})
        for name, entry in schedulers.items():
            summary = entry.get("summary", {})
            for field in REQUIRED_NUMERIC_FIELDS:
                self.assertIn(field, summary, f"Scheduler {name} missing summary field '{field}'")
                val = summary[field]
                self.assertNotIsInstance(
                    val, bool, f"Scheduler {name} field '{field}' must not be bool"
                )
                self.assertTrue(
                    isinstance(val, (int, float)),
                    f"Scheduler {name} field '{field}' must be numeric, got {type(val)}: {val}",
                )
                self.assertTrue(
                    math.isfinite(val),
                    f"Scheduler {name} field '{field}' must be finite, got {val}",
                )

    def test_benchmark_rejects_placeholder_pd(self):
        schedulers = self.data.get("schedulers", {})
        for name, entry in schedulers.items():
            pd = entry.get("summary", {}).get("pd")
            self.assertNotEqual(pd, "N/A", f"Scheduler {name} has placeholder pd 'N/A'")
            self.assertNotEqual(pd, "null", f"Scheduler {name} has string 'null' for pd")
            self.assertIsNotNone(pd, f"Scheduler {name} has null pd")
            self.assertTrue(0.0 <= pd <= 1.0, f"Scheduler {name} pd out of [0, 1]: {pd}")

    def test_benchmark_has_tp_fn_fp_tn(self):
        schedulers = self.data.get("schedulers", {})
        for name, entry in schedulers.items():
            summary = entry.get("summary", {})
            for counter in ["tp", "fn", "fp", "tn"]:
                self.assertIn(counter, summary, f"Scheduler {name} missing '{counter}'")
                val = summary[counter]
                self.assertIsInstance(val, int, f"Scheduler {name} '{counter}' must be int, got {type(val)}")
                self.assertGreaterEqual(val, 0, f"Scheduler {name} '{counter}' must be >= 0")

            # Check scenario breakdowns too
            breakdown = entry.get("scenario_breakdown", {})
            self.assertTrue(len(breakdown) > 0, f"Scheduler {name} must have scenario breakdown")
            for scen_name, scen_data in breakdown.items():
                for counter in ["tp", "fn", "fp", "tn"]:
                    self.assertIn(
                        counter, scen_data, f"Scheduler {name} scenario {scen_name} missing '{counter}'"
                    )

    def test_benchmark_pd_matches_tp_fn(self):
        schedulers = self.data.get("schedulers", {})
        for name, entry in schedulers.items():
            summary = entry.get("summary", {})
            tp = summary["tp"]
            fn = summary["fn"]
            pd = summary["pd"]
            expected_pd = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            self.assertAlmostEqual(
                pd,
                expected_pd,
                places=5,
                msg=f"Scheduler {name} summary pd ({pd}) does not match tp/fn expected ({expected_pd})",
            )

    def test_benchmark_pfa_matches_fp_tn(self):
        schedulers = self.data.get("schedulers", {})
        for name, entry in schedulers.items():
            summary = entry.get("summary", {})
            fp = summary["fp"]
            tn = summary["tn"]
            pfa = summary["pfa"]
            expected_pfa = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            self.assertAlmostEqual(
                pfa,
                expected_pfa,
                places=5,
                msg=f"Scheduler {name} summary pfa ({pfa}) does not match fp/tn expected ({expected_pfa})",
            )

    def test_benchmark_json_is_finite(self):
        raw_text = BENCHMARK_PATH.read_text(encoding="utf-8")
        self.assertNotIn("NaN", raw_text)
        self.assertNotIn("Infinity", raw_text)
        self.assertNotIn("-Infinity", raw_text)

    def test_api_benchmark_matches_authoritative_artifact(self):
        with TestClient(app) as client:
            res = client.get("/api/benchmark")
            self.assertEqual(res.status_code, 200)
            served_data = res.json()
            self.assertEqual(len(served_data["schedulers"]), 4)
            self.assertEqual(
                set(served_data["schedulers"].keys()), EXPECTED_SCHEDULERS
            )
            # Verify SmartScan_DRQN_MoE metrics match
            moe_summary = served_data["schedulers"]["SmartScan_DRQN_MoE"]["summary"]
            local_summary = self.data["schedulers"]["SmartScan_DRQN_MoE"]["summary"]
            self.assertAlmostEqual(moe_summary["pd"], local_summary["pd"], places=5)
            self.assertAlmostEqual(
                moe_summary["avg_reward"], local_summary["avg_reward"], places=5
            )

    def test_api_benchmark_fails_closed_on_corrupt_json(self):
        from fastapi import HTTPException

        # 1. Test missing schedulers
        with self.assertRaises(HTTPException) as ctx:
            validate_benchmark_payload({"metadata": {}})
        self.assertEqual(ctx.exception.status_code, 500)

        # 2. Test wrong number of schedulers
        with self.assertRaises(HTTPException) as ctx:
            validate_benchmark_payload(
                {"schedulers": {"SmartScan_DRQN_MoE": {"summary": {}}}}
            )
        self.assertEqual(ctx.exception.status_code, 500)

        # 3. Test string placeholder instead of numeric
        corrupt_payload = {
            "metadata": {"schema_version": "2026.1-CANONICAL"},
            "schedulers": {
                name: {
                    "summary": {
                        "pd": "N/A" if name == "SmartScan_DRQN_MoE" else 0.5,
                        "pfa": 0.0,
                        "sensitivity_dbm": -110.0,
                        "avg_intercept_rate": 0.1,
                        "avg_reward": -1.0,
                        "tp": 10,
                        "fn": 10,
                        "fp": 0,
                        "tn": 100,
                    }
                }
                for name in EXPECTED_SCHEDULERS
            },
        }
        with self.assertRaises(HTTPException) as ctx:
            validate_benchmark_payload(corrupt_payload)
        self.assertEqual(ctx.exception.status_code, 500)

        # 4. Test when file contains corrupt unparseable JSON
        with patch.object(Path, "read_text", return_value="{ this is corrupt json"):
            with TestClient(app) as client:
                res = client.get("/api/benchmark")
                self.assertEqual(res.status_code, 500)

        # 5. Test when file is missing in API endpoint
        with patch("ew_core.deployment.api.BENCHMARK_RESULTS_PATH", Path("non_existent_file.json")), \
             patch("ew_core.deployment.api.PACKAGE_ROOT", Path("non_existent_dir")):
            with TestClient(app) as client:
                res = client.get("/api/benchmark")
                self.assertEqual(res.status_code, 500)

    def test_api_health_and_ready_metadata(self):
        with TestClient(app) as client:
            h_res = client.get("/health")
            self.assertEqual(h_res.status_code, 200)
            h_data = h_res.json()
            self.assertIn("benchmark_artifact_sha256", h_data)
            self.assertIn("benchmark_schema_version", h_data)

            r_res = client.get("/ready")
            self.assertEqual(r_res.status_code, 200)
            r_data = r_res.json()
            self.assertIn("benchmark_artifact_sha256", r_data)
            self.assertIn("benchmark_schema_version", r_data)


if __name__ == "__main__":
    unittest.main()

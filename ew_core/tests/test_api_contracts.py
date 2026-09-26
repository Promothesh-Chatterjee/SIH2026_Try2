"""Regression tests for frontend-backend API contracts.

Verifies that every route consumed by frontend/src/services/api.js is registered
and responds with valid HTTP codes and JSON schemas on ew_core/deployment/api.py.
"""

from __future__ import annotations

import os
import pytest
from starlette.testclient import TestClient

from ew_core.deployment.api import app


@pytest.fixture
def client(monkeypatch):
    # Configure demo key for testing authenticated routes
    test_key = "test-api-key-12345"
    monkeypatch.setenv("SMARTSCAN_API_KEY", test_key)
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_health_endpoint(client: TestClient):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data


def test_metrics_endpoints(client: TestClient):
    # Prometheus/summary metrics
    res1 = client.get("/metrics")
    assert res1.status_code == 200

    # Evaluation dashboard metrics
    res2 = client.get("/api/v1/metrics")
    assert res2.status_code == 200
    data2 = res2.json()
    assert "pd" in data2
    assert "pfa" in data2

    # Spectrum data
    res3 = client.get("/api/v1/spectrum")
    assert res3.status_code == 200
    data3 = res3.json()
    assert "status" in data3


def test_telemetry_endpoints(client: TestClient):
    res1 = client.get("/telemetry/latest")
    assert res1.status_code == 200

    res2 = client.get("/telemetry/history?limit=10")
    assert res2.status_code == 200

    res3 = client.get("/telemetry/runs")
    assert res3.status_code == 200


def test_mission_lifecycle_endpoints(client: TestClient):
    headers = {"X-SmartScan-API-Key": "test-api-key-12345"}

    # Reset
    res_reset = client.post("/reset", headers=headers)
    assert res_reset.status_code == 200

    # Start
    res_start = client.post("/mission/start", json={"initial_time_us": 0.0}, headers=headers)
    assert res_start.status_code == 200

    # Status
    res_status = client.get("/mission/status")
    assert res_status.status_code == 200

    # Step (with dummy observation)
    res_step = client.post("/mission/step", json={"pdws": None, "obs": None}, headers=headers)
    assert res_step.status_code == 200

    # Stop
    res_stop = client.post("/mission/stop", headers=headers)
    assert res_stop.status_code == 200


def test_stream_status_endpoint(client: TestClient):
    res = client.get("/mission/stream/status")
    assert res.status_code == 200
    data = res.json()
    assert "running" in data


def test_memory_emitters_endpoint(client: TestClient):
    res = client.get("/memory/emitters")
    assert res.status_code == 200


def test_benchmark_endpoints(client: TestClient):
    res1 = client.get("/benchmark/latest")
    assert res1.status_code == 200

    res2 = client.get("/benchmark/scenarios")
    assert res2.status_code == 200

    res3 = client.get("/gnu_rf/scenarios")
    assert res3.status_code == 200


def test_predict_bands_validation(client: TestClient):
    headers = {"X-SmartScan-API-Key": "test-api-key-12345"}

    # Reject non-360 observation
    bad_obs = [0.0] * 10
    res = client.post("/predict_bands", json={"obs": bad_obs}, headers=headers)
    assert res.status_code in (422, 400)

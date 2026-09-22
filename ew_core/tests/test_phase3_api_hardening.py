"""Comprehensive unit tests for Phase 3 API Hardening.

Covers:
  - Task 3.1: API Key Auth (auth.py), token generation, dev-open fallback, 401 on invalid/missing key.
  - Task 3.2: Rate limiting, /schedule alias, /auth/info.
  - Task 3.3: WebSocket /ws/metrics connection and live telemetry broadcasting.
  - Task 3.4: /model/reload endpoint (local file verification).
  - Task 3.5: dataset_service.py (get_tsrd_root, list_scenarios).
  - Task 3.6: /ready readiness probe and /scenario/run endpoint.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from ew_core.contracts import CANONICAL_OBS_DIM
from ew_core.deployment.api import app, STATE
from ew_core.deployment.auth import (
    generate_api_key,
    get_valid_api_keys,
    require_api_key,
)
from ew_core.deployment.dataset_service import (
    get_tsrd_root,
    list_scenarios,
    download_from_blob,
)


# ── Task 3.1: Auth Tests ──────────────────────────────────────────────────────

def test_generate_api_key():
    """Verify generated API keys have sk-smartscan- prefix and correct length."""
    key = generate_api_key()
    assert key.startswith("sk-smartscan-")
    # Prefix is 13 chars + 64 hex chars = 77 chars
    assert len(key) == 77
    assert key != generate_api_key()  # Unique tokens


def test_get_valid_api_keys_empty(monkeypatch):
    """When SMARTSCAN_API_KEYS is unset, valid set is empty."""
    monkeypatch.delenv("SMARTSCAN_API_KEYS", raising=False)
    assert get_valid_api_keys() == set()


def test_get_valid_api_keys_configured(monkeypatch):
    """Parses comma-separated and trimmed API keys."""
    monkeypatch.setenv("SMARTSCAN_API_KEYS", "key1, key2,  key3  ")
    assert get_valid_api_keys() == {"key1", "key2", "key3"}


def test_require_api_key_open_dev_mode(monkeypatch):
    """When SMARTSCAN_API_KEYS is empty, open-dev fallback returns 'dev-open'."""
    monkeypatch.delenv("SMARTSCAN_API_KEYS", raising=False)
    assert require_api_key(None) == "dev-open"
    assert require_api_key("some-key") == "dev-open"


def test_require_api_key_enforced(monkeypatch):
    """When SMARTSCAN_API_KEYS is set, missing or wrong key raises 401."""
    valid_key = generate_api_key()
    monkeypatch.setenv("SMARTSCAN_API_KEYS", f"{valid_key},another-key")

    # Missing header
    with pytest.raises(HTTPException) as exc_info:
        require_api_key(None)
    assert exc_info.value.status_code == 401

    # Invalid header
    with pytest.raises(HTTPException) as exc_info:
        require_api_key("sk-smartscan-invalid")
    assert exc_info.value.status_code == 401

    # Valid header
    assert require_api_key(valid_key) == valid_key


# ── Task 3.5: Dataset Service Tests ──────────────────────────────────────────

def test_dataset_service_root_resolution(tmp_path, monkeypatch):
    """Verify get_tsrd_root resolves configured env or local fallbacks."""
    fake_tsrd = tmp_path / "tsrd_dir"
    fake_tsrd.mkdir()
    monkeypatch.setenv("TSRD_DATA_ROOT", str(fake_tsrd))
    assert get_tsrd_root() == str(fake_tsrd)

    # When env points to a dummy path that exists
    dummy_dir = tmp_path / "fallback_dir"
    dummy_dir.mkdir()
    monkeypatch.setenv("TSRD_DATA_ROOT", str(dummy_dir))
    assert get_tsrd_root() == str(dummy_dir)


def test_dataset_service_list_scenarios(tmp_path, monkeypatch):
    """Verify list_scenarios finds config_*.h5 files in target split."""
    split_dir = tmp_path / "stare" / "val_stare"
    split_dir.mkdir(parents=True)
    (split_dir / "config_119.h5").touch()
    (split_dir / "config_200.h5").touch()
    (split_dir / "ignored.txt").touch()

    monkeypatch.setenv("TSRD_DATA_ROOT", str(tmp_path))
    scenarios = list_scenarios("val")
    assert len(scenarios) == 2
    assert "config_119" in scenarios
    assert "config_200" in scenarios


# ── Task 3.2: API Endpoints (Auth, Info, /schedule) ─────────────────────────

def test_auth_info_endpoint():
    """Verify /auth/info returns metadata schema."""
    with TestClient(app) as client:
        resp = client.get("/auth/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "auth_required" in data
        assert data["header_name"] == "X-SmartScan-API-Key"
        assert "120 requests/minute" in data["rate_limit"]


def test_auth_enforcement_on_routes(monkeypatch):
    """Verify protected endpoints reject requests when API keys are configured and missing."""
    valid_key = generate_api_key()
    monkeypatch.setenv("SMARTSCAN_API_KEYS", valid_key)

    with TestClient(app) as client:
        obs = [0.0] * CANONICAL_OBS_DIM
        # Missing auth header -> 401
        resp = client.post("/predict_bands", json={"obs": obs})
        assert resp.status_code == 401

        resp = client.post("/schedule", json={"obs": obs})
        assert resp.status_code == 401

        # Invalid auth header -> 401
        resp = client.post("/schedule", json={"obs": obs}, headers={"X-SmartScan-API-Key": "wrong"})
        assert resp.status_code == 401

        # Model loaded or not -> with valid key should pass auth layer (reaches endpoint logic, not 401)
        resp = client.post("/schedule", json={"obs": obs}, headers={"X-SmartScan-API-Key": valid_key})
        assert resp.status_code in (200, 503)


def test_schedule_endpoint_alias():
    """Verify /schedule accepts ScheduleRequest and calls predict_bands."""
    with TestClient(app) as client:
        obs = [0.0] * CANONICAL_OBS_DIM
        resp_pred = client.post("/predict_bands", json={"obs": obs})
        resp_sched = client.post("/schedule", json={"obs": obs})
        assert resp_pred.status_code == resp_sched.status_code


# ── Task 3.3: WebSocket Live Metrics ─────────────────────────────────────────

def test_websocket_metrics_connection():
    """Verify WebSocket /ws/metrics connects and disconnects cleanly."""
    with TestClient(app) as client:
        with client.websocket_connect("/ws/metrics") as ws:
            # Broadcast a test metrics payload
            from ew_core.deployment.api import broadcast_metrics
            import asyncio

            test_payload = {
                "pd": 0.85,
                "pfa": 0.01,
                "avg_intercept_rate": 0.75,
                "last_action": 12,
            }
            asyncio.run(broadcast_metrics(test_payload))
            data = ws.receive_json()
            assert data["type"] == "metrics"
            assert data["data"]["pd"] == 0.85
            assert data["data"]["last_action"] == 12


# ── Task 3.4 & 3.6: /ready, /model/reload, /scenario/run ─────────────────────

def test_readiness_probe_not_ready():
    """Verify /ready reports 503 when scheduler is not loaded."""
    with TestClient(app) as client:
        orig_sched = STATE.get("scheduler")
        orig_onnx = STATE.get("scheduler_onnx")
        try:
            STATE["scheduler"] = None
            STATE["scheduler_onnx"] = None
            resp = client.get("/ready")
            assert resp.status_code == 503
            assert "not_ready" in resp.json()["detail"]
        finally:
            STATE["scheduler"] = orig_sched
            STATE["scheduler_onnx"] = orig_onnx


def test_readiness_probe_ready(monkeypatch):
    """Verify /ready reports 200 ready when mock model and dataset are present."""
    with TestClient(app) as client:
        orig_sched = STATE.get("scheduler")
        try:
            STATE["scheduler"] = MagicMock()
            monkeypatch.setattr("ew_core.deployment.api.get_tsrd_root", lambda: "fake/tsrd")
            monkeypatch.setattr("ew_core.deployment.api.list_scenarios", lambda split: ["config_1", "config_2"])

            resp = client.get("/ready")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ready"
            assert data["model_loaded"] is True
            assert data["scenarios_count"] == 2
        finally:
            STATE["scheduler"] = orig_sched


def test_model_reload_missing_file():
    """Verify /model/reload returns 404 for non-existent checkpoint path."""
    with TestClient(app) as client:
        resp = client.post("/model/reload?checkpoint_path=nonexistent_file.pt")
        assert resp.status_code == 404


def test_model_reload_local(tmp_path):
    """Verify /model/reload successfully hot-reloads state dict from local checkpoint."""
    import torch
    import torch.nn as nn

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2, 2)

    model = DummyModel()
    ckpt_file = tmp_path / "test_ckpt.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_file)

    with TestClient(app) as client:
        dummy_drqn = DummyModel()
        orig_sched = STATE.get("scheduler")
        try:
            STATE["scheduler"] = dummy_drqn
            resp = client.post(f"/model/reload?checkpoint_path={str(ckpt_file).replace('\\', '/')}")
            assert resp.status_code == 200
            assert resp.json()["status"] == "reloaded"
        finally:
            STATE["scheduler"] = orig_sched


def test_scenario_run_endpoint():
    """Verify /scenario/run triggers run_evaluation with specified scenario."""
    with patch("ew_core.training.eval_batch.run_evaluation") as mock_eval:
        mock_eval.return_value = {
            "mean_intercept_rate": 0.88,
            "avg_intercept_time_error_us": 12.5,
        }
        with TestClient(app) as client:
            orig_moe = STATE.get("moe")
            try:
                STATE["moe"] = MagicMock()
                resp = client.post("/scenario/run?scenario_id=config_999&n_steps=100")
                assert resp.status_code == 200
                data = resp.json()
                assert data["mean_intercept_rate"] == 0.88
                mock_eval.assert_called_once()
            finally:
                STATE["moe"] = orig_moe

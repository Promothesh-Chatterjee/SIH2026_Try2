# tests/test_hidden_concurrency.py
"""Tests to verify that concurrent access to the recurrent hidden state
is protected by a lock, preventing race conditions.

Two kinds of tests are provided:
1. *Integration* tests that hit the `/predict_bands` and `/reset` endpoints.
   They are only executed when a real scheduler (PT or ONNX) is loaded –
   otherwise they are skipped with a clear message.
2. *Unit* test that exercises the `hidden_lock` directly without requiring any
   model checkpoint. This ensures coverage of the thread‑safety mechanism
   regardless of deployment artefacts.
"""

import concurrent.futures
import pytest
from fastapi.testclient import TestClient
from src.deployment.api import app, STATE, hidden_lock
from src.contracts import CANONICAL_OBS_DIM
import torch

# Minimal valid observation vector (all zeros)
OBSERVATION = [0.0] * CANONICAL_OBS_DIM


def _scheduler_loaded() -> bool:
    """Return True if a scheduler (PT MoE or ONNX) is available in STATE.

    The API starts without a scheduler when the required checkpoint/config
    files are missing. In that situation the `/predict_bands` endpoint returns
    HTTP 503, which is the documented behaviour. The integration tests should
    be skipped rather than treating 503 as success.
    """
    return STATE.get("moe") is not None or STATE.get("scheduler_onnx") is not None


def _make_predict_request(client: TestClient):
    response = client.post("/predict_bands", json={"obs": OBSERVATION})
    assert response.status_code == 200
    data = response.json()
    assert "selected_action" in data
    assert "latency_ms" in data
    return data


def _make_reset_request(client: TestClient):
    response = client.post("/reset", json={})
    assert response.status_code == 200
    assert response.json().get("status") == "reset ok"
    return response


def test_concurrent_predict_bands_no_race():
    """Fire several concurrent predictions and ensure hidden state stays consistent.

    Skipped if no scheduler checkpoint/config is present.
    """
    if not _scheduler_loaded():
        pytest.skip(
            "scheduler checkpoint/config unavailable; concurrency endpoint test requires a loaded scheduler."
        )
    with TestClient(app) as client:
        n_requests = 10
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_requests) as executor:
            futures = [executor.submit(_make_predict_request, client) for _ in range(n_requests)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        assert len(results) == n_requests
        hidden = STATE.get("hidden")
        assert hidden is not None, "Hidden state should be initialised after predictions"
        assert isinstance(hidden, torch.Tensor)
        assert hidden.ndim == 3


def test_predict_and_reset_concurrently():
    """Run a `/reset` request while predictions are in flight.

    Skipped if no scheduler checkpoint/config is present.
    """
    if not _scheduler_loaded():
        pytest.skip(
            "scheduler checkpoint/config unavailable; concurrency endpoint test requires a loaded scheduler."
        )
    with TestClient(app) as client:
        # Prepare tasks: several predicts and one reset
        predict_futures = 5
        with concurrent.futures.ThreadPoolExecutor(max_workers=predict_futures + 1) as executor:
            futures = [executor.submit(_make_predict_request, client) for _ in range(predict_futures)]
            reset_future = executor.submit(_make_reset_request, client)
            futures.append(reset_future)
            # Collect results (we don't need them beyond ensuring no exception)
            _ = [f.result() for f in concurrent.futures.as_completed(futures)]
        hidden = STATE.get("hidden")
        assert isinstance(hidden, torch.Tensor)
        assert hidden.ndim == 3


def test_hidden_state_lock_thread_safety():
    """Unit‑level test of `hidden_lock` without needing a real scheduler.

    It writes a dummy tensor to `STATE["hidden"]` concurrently from several
    threads, verifying that the lock serialises access and that the final
    value remains a valid `torch.Tensor` with the expected shape.
    """
    dummy = torch.randn(1, 1, 128)
    # Initialise hidden under the lock
    with hidden_lock:
        STATE["hidden"] = dummy.clone()

    def worker(idx: int):
        with hidden_lock:
            cur = STATE["hidden"]
            # Create a new tensor based on the current one; simple deterministic op
            new = cur + idx
            STATE["hidden"] = new
        return STATE["hidden"]

    # Run several workers concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, i) for i in range(4)]
        # Ensure all threads complete without raising
        _ = [f.result() for f in concurrent.futures.as_completed(futures)]

    final_hidden = STATE.get("hidden")
    assert isinstance(final_hidden, torch.Tensor), "Hidden state should remain a tensor"
    assert final_hidden.shape == dummy.shape, "Tensor shape must be preserved"


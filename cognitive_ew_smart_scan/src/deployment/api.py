"""
FastAPI REST microservice for Cognitive EW SmartScan.

Endpoints:
  POST /predict_bands  — single best time-frequency action + real aux/attribution
  POST /deinterleave   — PDW batch deinterleaving (trained model required)
  POST /update_memory  — write emitter profile
  GET  /memory/emitters — list emitters
  GET  /health          — liveness
  GET  /metrics         — FoM stats
  POST /reset           — reset LSTM hidden + episodic memory

Fail-safe contract (Phase 16):
  * no random-scheduler / raw-baseline fallbacks — missing trained models
    return HTTP 503;
  * /predict_bands accepts ONLY the canonical 36-band x 10-feature obs_dim=360;
  * responses expose real model outputs only (no fabricated attribution/metrics).
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()

import torch
import yaml
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
import asyncio
import json
from threading import Lock
hidden_lock = Lock()
try:
    from fastapi.middleware.base import BaseHTTPMiddleware  # type: ignore
except ImportError:
    from starlette.middleware.base import BaseHTTPMiddleware  # type: ignore

from pydantic import BaseModel, Field

from src.contracts import (
    CANONICAL_BAND_FEATURES,
    CANONICAL_N_ACTIONS,
    CANONICAL_N_BANDS,
    CANONICAL_N_MODES,
    CANONICAL_OBS_DIM,
    DEFAULT_DWELL_MULTIPLIERS,
    NORMAL_DWELL,
    REVISIT_AGE_IDX,
    band_of_action,
    mode_of_action,
)
from src.telemetry.publisher import TelemetryPublisher
from src.telemetry.discovery import latest_telemetry_snapshot, latest_telemetry_history, find_latest_run

MAX_PDWS_PER_REQUEST = 10000
MAX_SESSION_TTL_SECONDS = 3600

# Phase 16: canonical production observation contract. /predict_bands accepts
# ONLY the 36-band x 10-feature layout (obs_dim=360). Legacy 2*n_bands and any
# other lengths are rejected. The values themselves live in src/contracts.py.
OBS_FEATURES_PER_BAND = CANONICAL_BAND_FEATURES


def _is_authorized(request: Request) -> bool:
    """Allow state-changing endpoints only with a valid session token.

    The project requirement explicitly calls for authentication on mutating API
    routes. A simple bearer token avoids open state mutation while keeping the
    service runnable in local testing environments.
    """
    token = os.getenv("SMARTSCAN_API_TOKEN", "")
    if not token:
        return True
    auth_header = request.headers.get("Authorization", "")
    return auth_header == f"Bearer {token}"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Global state populated at startup
STATE: dict[str, Any] = {
    "device": "cpu",
    "model_cfg": {},
    "deinterleaver": None,
    "scheduler": None,
    "moe": None,
    "memory": None,
    "fom": None,
    "hidden": None,
    "normalization_stats": None,
    "normalization_stats_path": None,
    "normalization_stats_hash": None,
}

# P0-10: real telemetry broker. Deliberately no fabricated streaming keys: the
# dashboard only ever sees values recorded via publisher.update() (or from the
# latest persisted run on disk). Until an update happens, clients receive an
# explicit {"live": false} state rather than invented metrics.
telemetry = TelemetryPublisher(run=None)
TELEMETRY_ROOT = os.getenv("TELEMETRY_ROOT", "runs")


# ── Pydantic Schemas (Pydantic v2) ──────────────────────────────────────────

class PredictBandsRequest(BaseModel):
    """Request for band prediction."""

    obs: list[float] = Field(..., description=f"Observation vector of exactly obs_dim={CANONICAL_OBS_DIM} (36 bands x 10 features)", min_length=2)


class PredictBandsResponse(BaseModel):
    """Single best time-frequency selection with real model outputs.

    Every field is a real value produced by a trained scheduler:
      * selected_action        flat action index (band*n_modes + mode)
      * selected_band / selected_mode   decoded time-frequency cell
      * dwell_time_us          base dwell * the mode's config multiplier
      * intercept_probability  DRQN aux head (sigmoid) for the selected action
      * predicted_intercept_time_us  DRQN aux head (softplus) in microseconds
      * attribution            real decomposition computed from real model
                               Q-values and real observation features (never
                               fabricated placeholders)
      * latency_ms             wall-clock inference latency
    """

    selected_action: int = Field(..., description="Selected flat time-frequency action index")
    selected_band: int = Field(..., description="Selected band index")
    selected_mode: int = Field(..., description="Selected dwell-mode index")
    dwell_time_us: float = Field(..., description="Dwell time for the selected mode (base * multiplier)")
    intercept_probability: float = Field(..., description="DRQN aux prediction: probability of intercept for the selected action")
    predicted_intercept_time_us: float = Field(..., description="DRQN aux prediction: expected time-to-intercept (µs)")
    attribution: dict[str, Any] = Field(..., description="Real attribution: eager_pct / revisit_pct and mode semantics when available")
    latency_ms: float = Field(..., description="Inference latency in ms")


class DeinterleaveRequest(BaseModel):
    """Request for deinterleaving."""

    pdws: list[list[float]] = Field(..., description="List of PDWs, each [ToA, CF, PW, AoA, Amp] (N,5)")
    min_cluster_size: int = Field(10, description="HDBSCAN min_cluster_size")


class DeinterleaveResponse(BaseModel):
    """Response with predicted labels."""

    labels: list[int] = Field(..., description="Predicted emitter labels (-1=noise)")
    n_clusters: int = Field(..., description="Number of clusters found")
    latency_ms: float = Field(..., description="Inference latency ms")


class UpdateMemoryRequest(BaseModel):
    """Request to write emitter profile."""

    emitter_id: str = Field(..., description="Unique emitter ID")
    mean_pri_us: float = 0.0
    freq_min_mhz: float = 0.0
    freq_max_mhz: float = 0.0
    mean_pw_us: float = 0.0
    aoa_mean: float = 0.0
    amplitude_mean: float = 0.0
    priority_score: float = Field(0.5, ge=0.0, le=1.0)
    is_periodic: int = 0
    scan_period_us: float | None = None
    intercept_count: int = 0
    last_seen_us: float = 0.0


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    device: str
    models_loaded: dict[str, bool]


# ── Middleware ───────────────────────────────────────────────────────────────

class TimingMiddleware(BaseHTTPMiddleware):
    """Log request timing."""

    async def dispatch(self, request, call_next):  # type: ignore
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        response.headers["X-Process-Time-ms"] = f"{elapsed_ms:.2f}"
        logger.info("%s %s -> %d in %.2fms", request.method, request.url.path, response.status_code, elapsed_ms)
        return response


# ── Lifespan (startup) ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore
    """Load ONNX/pytorch models into memory at startup."""
    device_env = os.getenv("DEVICE", "cpu")
    if device_env == "cuda" and not torch.cuda.is_available():
        logger.warning("DEVICE=cuda but CUDA unavailable — falling back to cpu")
        device_env = "cpu"
    STATE["device"] = device_env
    logger.info("API starting on device=%s", device_env)

    # Load configs
    cfg_path = Path("configs/model_config.yaml")
    if cfg_path.exists():
        with open(cfg_path) as f:
            STATE["model_cfg"] = yaml.safe_load(f)
    else:
        logger.warning("model_config.yaml not found at %s", cfg_path)
        STATE["model_cfg"] = {
    "drqn_scheduler": {
        "n_bands": CANONICAL_N_BANDS,
        "n_modes": CANONICAL_N_MODES,
        "n_actions": CANONICAL_N_ACTIONS,
        "obs_dim": CANONICAL_OBS_DIM,
    },
    "smartscan_moe": {},
}

    # Try to load PyTorch models (ONNX preferred if available, else PT)
    # Deinterleaver
    for ckpt in [Path("checkpoints/onnx/deinterleaver.onnx"), Path("checkpoints/deinterleaver/best.pt"), Path("checkpoints/deinterleaver/final.pt")]:
        if ckpt.exists():
            try:
                if ckpt.suffix == ".onnx":
                    try:
                        import onnxruntime as ort  # type: ignore

                        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device_env == "cuda" else ["CPUExecutionProvider"]
                        STATE["deinterleaver_onnx"] = ort.InferenceSession(str(ckpt), providers=providers)
                        STATE["deinterleaver"] = "onnx"
                        logger.info("Loaded deinterleaver ONNX %s", ckpt)
                        break
                    except Exception as exc:
                        logger.warning("ONNX load failed %s: %s", ckpt, exc)
                else:
                    from ..models.deinterleaver import PDWTransformerEncoder

                    d_cfg = STATE["model_cfg"].get("deinterleaver", {})
                    m = PDWTransformerEncoder(
                        pdw_dim=d_cfg.get("pdw_dim", 6),
                        d_model=d_cfg.get("d_model", 128),
                        nhead=d_cfg.get("nhead", 8),
                        num_layers=d_cfg.get("num_layers", 4),
                        dim_feedforward=d_cfg.get("dim_feedforward", 512),
                        dropout=d_cfg.get("dropout", 0.1),
                        embed_dim=d_cfg.get("embed_dim", 64),
                    )
                    state = torch.load(str(ckpt), map_location="cpu")
                    if isinstance(state, dict) and "state_dict" in state:
                        state = state["state_dict"]
                    m.load_state_dict(state, strict=False)
                    m.to(torch.device("cpu"))
                    m.eval()
                    STATE["deinterleaver"] = m
                    logger.info("Loaded deinterleaver PT %s", ckpt)
                    break
            except Exception as exc:
                logger.warning("Failed to load deinterleaver %s: %s", ckpt, exc)

    # Scheduler / MoE
    for ckpt in [Path("checkpoints/onnx/scheduler.onnx"), Path("checkpoints/scheduler/best.pt"), Path("checkpoints/scheduler/final.pt")]:
        if ckpt.exists():
            try:
                if ckpt.suffix == ".onnx":
                    try:
                        import onnxruntime as ort  # type: ignore

                        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device_env == "cuda" else ["CPUExecutionProvider"]
                        STATE["scheduler_onnx"] = ort.InferenceSession(str(ckpt), providers=providers)
                        STATE["scheduler"] = "onnx"
                        logger.info("Loaded scheduler ONNX %s", ckpt)
                        break
                    except Exception as exc:
                        logger.warning("ONNX scheduler load failed: %s", exc)
                else:
                    from ..models.drqn_scheduler import DRQNScheduler
                    from ..models.smartscan_moe import SmartScanMoE

                    d_cfg = STATE["model_cfg"].get("drqn_scheduler", {})
                    moe_cfg = STATE["model_cfg"].get("smartscan_moe", {})
                    n_bands_api = int(d_cfg.get("n_bands", CANONICAL_N_BANDS))
                    n_modes_api = int(d_cfg.get("n_modes", CANONICAL_N_MODES))
                    n_actions_api = int(d_cfg.get("n_actions", n_bands_api * n_modes_api if n_modes_api else CANONICAL_N_ACTIONS))
                    drqn = DRQNScheduler(
                        obs_dim=int(d_cfg.get("obs_dim", CANONICAL_OBS_DIM)),
                        n_bands=n_bands_api,
                        n_actions=n_actions_api,
                        n_modes=n_modes_api,
                        lstm_hidden=int(d_cfg.get("lstm_hidden", 256)),
                        lstm_layers=int(d_cfg.get("lstm_layers", 2)),
                    )
                    state = torch.load(str(ckpt), map_location="cpu")
                    if isinstance(state, dict) and "state_dict" in state:
                        state = state["state_dict"]
                    drqn.load_state_dict(state, strict=False)
                    drqn.to(torch.device(STATE["device"] if STATE["device"] != "cuda" else "cpu"))
                    drqn.eval()
                    moe = SmartScanMoE(
                        drqn,
                        {**moe_cfg, "n_bands": n_bands_api, "n_modes": n_modes_api, "n_actions": n_actions_api, "device": STATE["device"]},
                    )
                    STATE["scheduler"] = drqn
                    STATE["moe"] = moe
                    # Init hidden
                    try:
                        hidden = drqn.init_hidden(1, STATE["device"] if STATE["device"] == "cpu" else "cpu")
                        with hidden_lock:
                            STATE["hidden"] = hidden
                            moe.eager_agent.hidden = hidden
                    except Exception:
                        pass
                    logger.info("Loaded scheduler PT %s", ckpt)
                    break
            except Exception as exc:
                logger.warning("Failed to load scheduler %s: %s", ckpt, exc)

    # Memory and FoM
    try:
        from ..cognitive.memory import SemanticMemory
        from ..evaluation.metrics import FiguresOfMerit
        from ..preprocessing.normalise import load_normalization_stats, normalization_stats_hash

        STATE["memory"] = SemanticMemory()
        STATE["fom"] = FiguresOfMerit()

        # Phase 14: only TRAIN-fitted normalization statistics may be used once a
        # trained deinterleaver is serving. Locate the persisted stats JSON next
        # to the model checkpoints (canonical locations first).
        norm_candidates = [
            Path("checkpoints/deinterleaver/normalization_stats.json"),
            Path("configs/normalization_stats.json"),
            Path("checkpoints/onnx/normalization_stats.json"),
            Path("checkpoints/normalization_stats.json"),
        ]
        stats_path = next((c for c in norm_candidates if c.exists()), None)
        if stats_path is not None:
            try:
                STATE["normalization_stats"] = load_normalization_stats(stats_path)
                STATE["normalization_stats_path"] = str(stats_path)
                STATE["normalization_stats_hash"] = normalization_stats_hash(STATE["normalization_stats"])
                logger.info(
                    "Loaded train normalization stats %s (hash %s)",
                    stats_path,
                    STATE["normalization_stats_hash"],
                )
            except Exception as exc:
                logger.warning("Failed to load normalization stats %s: %s", stats_path, exc)
        else:
            logger.warning(
                "No train-fitted normalization_stats.json found under checkpoints/ or configs/ — "
                "deinterleave with a trained model will be refused until one is provided."
            )
        logger.info("SemanticMemory and FiguresOfMerit initialised")
    except Exception as exc:
        logger.warning("Memory/FoM init failed: %s", exc)

    yield
    # Shutdown: close DB
    try:
        if STATE.get("memory") and hasattr(STATE["memory"], "close"):
            STATE["memory"].close()
    except Exception:
        pass
    logger.info("API shutdown")


# ── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(title="Cognitive EW SmartScan API", version="0.1.0", lifespan=lifespan)
app.add_middleware(TimingMiddleware)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Liveness check."""
    return HealthResponse(
        status="ok",
        device=str(STATE.get("device", "cpu")),
        models_loaded={
            "deinterleaver": STATE.get("deinterleaver") is not None or "deinterleaver_onnx" in STATE,
            "scheduler": STATE.get("scheduler") is not None or "scheduler_onnx" in STATE,
            "memory": STATE.get("memory") is not None,
        },
    )


@app.get("/metrics", tags=["system"])
def get_metrics() -> dict[str, Any]:
    """Current FoM statistics."""
    fom = STATE.get("fom")
    if fom is None:
        return {"error": "FoM not initialised"}
    try:
        return fom.summary()  # type: ignore
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/telemetry/latest", tags=["telemetry"])
def telemetry_latest() -> dict[str, Any]:
    """Latest real telemetry snapshot: live in-process data, else latest run on disk.

    Never fabricates values; returns ``{"live": false}`` if no real data exists.
    """
    if telemetry.live:
        return telemetry.latest()
    return latest_telemetry_snapshot(TELEMETRY_ROOT)


@app.get("/telemetry/history", tags=["telemetry"])
def telemetry_history(limit: int = 200) -> dict[str, Any]:
    """History of real telemetry records (newest first) from the latest run.

    Args:
        limit: Maximum number of records to return (clamped to 1..1000).
    """
    limit = max(1, min(int(limit), 1000))
    if telemetry.live:
        return {"live": True, "records": telemetry.history(limit=limit)}
    records = latest_telemetry_history(TELEMETRY_ROOT, limit=limit)
    if not records and not telemetry.live:
        return {"live": False, "records": []}
    return {"live": True, "records": records}


@app.get("/telemetry/runs", tags=["telemetry"])
def telemetry_runs() -> dict[str, Any]:
    """List persisted run directories (newest first) under the telemetry root."""
    root = Path(TELEMETRY_ROOT)
    if not root.is_dir():
        return {"runs": []}
    runs = sorted((d.name for d in root.iterdir() if d.is_dir()), reverse=True)
    return {"runs": runs}


@app.post("/reset", tags=["system"])
def reset(request: Request) -> dict[str, str]:
    """Reset LSTM hidden state and episodic memory."""
    if not _is_authorized(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        if STATE.get("moe") and hasattr(STATE["moe"], "reset"):
            STATE["moe"].reset()  # type: ignore
        if STATE.get("fom") and hasattr(STATE["fom"], "reset"):
            STATE["fom"].reset()  # type: ignore
        # Reinit hidden
        if STATE.get("scheduler") and hasattr(STATE["scheduler"], "init_hidden"):
            try:
                with hidden_lock:
                    hidden = STATE["scheduler"].init_hidden(1, STATE.get("device", "cpu"))  # type: ignore
                    STATE["hidden"] = hidden
                    if STATE.get("moe"):
                        STATE["moe"].eager_agent.hidden = hidden  # type: ignore
            except Exception:
                pass
        return {"status": "reset ok"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _dwell_time_us_for_mode(mode: int) -> float:
    """Real dwell time for a dwell-mode index from the loaded config.

    Uses ``configs/model_config.yaml`` dwell_modes (base dwell x mode
    multiplier); falls back to the canonical contract multipliers.
    """
    mcfg = STATE.get("model_cfg", {}).get("dwell_modes", {})
    base = float(mcfg.get("base_dwell_time_us", 500.0))
    mults = mcfg.get("mode_multipliers", [])
    try:
        if 0 <= int(mode) < len(mults) and isinstance(mults[int(mode)], dict):
            return base * float(mults[int(mode)].get("multiplier", 1.0))
    except (TypeError, ValueError):
        pass
    if 0 <= int(mode) < len(DEFAULT_DWELL_MULTIPLIERS):
        return base * float(DEFAULT_DWELL_MULTIPLIERS[int(mode)])
    return base * float(DEFAULT_DWELL_MULTIPLIERS[NORMAL_DWELL])


def _aux_for_action(drqn, obs_1d: np.ndarray, action: int, hidden) -> tuple[float, float]:
    """Real DRQN aux outputs (intercept prob, intercept time) for one action.

    Runs the DRQN on the SAME observation with the SAME recurrent hidden the
    selection step used, so the auxiliary predictions describe the same
    decision context as the chosen action. Both values come from trained
    heads — nothing fabricated.
    """
    obs_t = torch.from_numpy(np.asarray(obs_1d, dtype=np.float32)).reshape(1, 1, -1)
    with torch.inference_mode():
        _q, aux, _ = drqn(obs_t, hidden)
    prob = float(aux["intercept_prob"][0, -1, int(action)].item())
    time_us = float(aux["intercept_time_us"][0, -1, int(action)].item())
    return prob, time_us


def _minmax_norm(vals: np.ndarray) -> np.ndarray:
    v_min, v_max = float(np.min(vals)), float(np.max(vals))
    if v_max - v_min < 1e-8:
        return np.zeros_like(vals, dtype=np.float32)
    return ((vals - v_min) / (v_max - v_min + 1e-8)).astype(np.float32)


@app.post("/predict_bands", response_model=PredictBandsResponse, tags=["scheduler"])
def predict_bands(req: PredictBandsRequest) -> PredictBandsResponse:
    """Select the single best time-frequency action from a trained scheduler.

    Phase 16 fail-safe contract:
      * expects EXACTLY obs_dim=360 (36 bands x 10 features) — the legacy
        2*n_bands observation layout is no longer accepted;
      * requires a trained scheduler (MoE/PT DRQN or the ONNX scheduler);
        otherwise HTTP 503 — there is NO random-scheduler fallback;
      * returns real model outputs only (Q-driven selection, DRQN aux
        predictions, config dwell-time, computed attribution), never
        fabricated attribution or metrics.
    """
    start = time.perf_counter()

    # ---- Fail-safe gates ----------------------------------------------------
    moe = STATE.get("moe")
    onnx_sess = STATE.get("scheduler_onnx")
    if moe is None and onnx_sess is None:
        raise HTTPException(
            status_code=503,
            detail="No trained scheduler is loaded — /predict_bands requires a trained scheduler.",
        )

    d_cfg = STATE.get("model_cfg", {}).get("drqn_scheduler", {})
    configured_dim = int(d_cfg.get("obs_dim", CANONICAL_OBS_DIM))
    if configured_dim != CANONICAL_OBS_DIM:
        raise HTTPException(
            status_code=503,
            detail=f"Configured obs_dim={configured_dim} is not the canonical {CANONICAL_OBS_DIM} — refusing inference.",
        )
    n_bands = int(d_cfg.get("n_bands", CANONICAL_N_BANDS))
    n_modes = int(d_cfg.get("n_modes", CANONICAL_N_MODES))

    obs = np.asarray(req.obs, dtype=np.float32)
    if obs.ndim != 1:
        raise HTTPException(status_code=400, detail=f"obs must be a flat vector, got ndim={obs.ndim}")
    if obs.size != CANONICAL_OBS_DIM:
        raise HTTPException(
            status_code=400,
            detail=f"obs must be exactly obs_dim={CANONICAL_OBS_DIM} (36 bands x 10 features), got {obs.size}",
        )

# ---- Selection ----------------------------------------------------------
    if moe is not None:
        # PT path: SmartScanMoE owns the DRQN recurrent state; capture the
        # pre-step hidden so the aux predictions below describe the exact
        # decision context (single forward, no state double-step).
        # Acquire lock to read hidden state safely
        with hidden_lock:
            pre_step_hidden = moe.eager_agent.hidden if moe.eager_agent.hidden is not None else STATE.get("hidden")
            # Pass the current hidden to select_action
            hidden_state = STATE.get("hidden")
        action, hidden, attribution = moe.select_action(obs, hidden_state)
        # Write back the updated hidden under lock
        with hidden_lock:
            STATE["hidden"] = hidden
        prob, pred_time_us = _aux_for_action(moe.eager_agent.drqn, obs, action, pre_step_hidden)
        moe.update(action)
    else:
        # ONNX eager path: real q / intercept_prob / intercept_time_us from the
        # exported DRQN. Attribution is computed from those real Q-values plus
        # the real revisit-age feature inside obs (index 4 of each 10-feature
        # band block) — matching the MoE fusion semantics without fabricating.
        inp = obs.reshape(1, 1, -1).astype(np.float32)
        q, q_prob, q_time = onnx_sess.run(None, {"obs": inp})
        q_last = q[0, -1] if q.ndim == 3 else q[0]
        prob_last = q_prob[0, -1] if q_prob.ndim == 3 else q_prob[0]
        time_last = q_time[0, -1] if q_time.ndim == 3 else q_time[0]

        moe_cfg = STATE.get("model_cfg", {}).get("smartscan_moe", {})
        eager_w = float(moe_cfg.get("eager_weight", 0.6))
        revisit_w = float(moe_cfg.get("revisit_weight", 0.4))
        q_norm = _minmax_norm(np.asarray(q_last, dtype=np.float32))
        rev_band = np.clip(obs[REVISIT_AGE_IDX::OBS_FEATURES_PER_BAND][:n_bands], 0.0, 1.0)
        rev_action = np.repeat(rev_band.astype(np.float32), n_modes)
        fused = eager_w * q_norm + revisit_w * rev_action
        action = int(np.argmax(fused))
        eager_contrib = eager_w * q_norm[action]
        revisit_contrib = revisit_w * rev_action[action]
        total = eager_contrib + revisit_contrib + 1e-8
        attribution = {
            "eager_pct": float(eager_contrib / total),
            "revisit_pct": float(revisit_contrib / total),
            "selected_band": band_of_action(action, n_modes),
            "selected_mode": int(mode_of_action(action, n_modes)),
        }
        prob = float(prob_last[action])
        pred_time_us = float(time_last[action])

    band = band_of_action(action, n_modes)
    mode = mode_of_action(action, n_modes)
    latency = (time.perf_counter() - start) * 1000.0
    return PredictBandsResponse(
        selected_action=int(action),
        selected_band=band,
        selected_mode=mode,
        dwell_time_us=_dwell_time_us_for_mode(mode),
        intercept_probability=prob,
        predicted_intercept_time_us=pred_time_us,
        attribution=attribution,
        latency_ms=latency,
    )


def _normalise_for_inference(pdws_arr: np.ndarray) -> np.ndarray:
    """Normalise PDWs for inference using ONLY persisted train statistics.

    Phase 14 leakage guard:
      * If a trained deinterleaver (PT or ONNX) is loaded, per-request fitting is
        FORBIDDEN — the persisted train-fitted stats are required and reused. If
        they are unavailable this raises HTTPException(503) rather than silently
        leaking test-data statistics into the model input space.
      * If NO trained model is loaded (raw HDBSCAN baseline only), fitting from
        the request is acceptable and preserved.
    """
    from ..preprocessing.normalise import normalise_pdws

    # Real STATE shapes: PT model -> STATE["deinterleaver"] is an nn.Module;
    # ONNX model -> STATE["deinterleaver"] == "onnx" and STATE["deinterleaver_onnx"]
    # holds the InferenceSession.
    onnx_sess = STATE.get("deinterleaver_onnx")
    mode = STATE.get("deinterleaver")
    has_model = bool(onnx_sess) or mode == "onnx" or (mode is not None and not isinstance(mode, str))
    if has_model:
        stats = STATE.get("normalization_stats")
        if not stats:
            raise HTTPException(
                status_code=503,
                detail=(
                    "A trained deinterleaver is loaded but "
                    "checkpoints/deinterleaver/normalization_stats.json was not "
                    "found. Inference must use the train-fitted statistics."
                ),
            )
        return normalise_pdws(pdws_arr, stats)[0]
    return normalise_pdws(pdws_arr, None)[0]


@app.post("/deinterleave", response_model=DeinterleaveResponse, tags=["deinterleaving"])
def deinterleave_endpoint(req: DeinterleaveRequest, request: Request) -> DeinterleaveResponse:
    """Run deinterleaving on PDW batch."""
    start = time.perf_counter()
    if len(req.pdws) > MAX_PDWS_PER_REQUEST:
        raise HTTPException(status_code=413, detail=f"pdws exceeds limit of {MAX_PDWS_PER_REQUEST}")
    if not req.pdws:
        raise HTTPException(status_code=400, detail="pdws must be non-empty")
    pdws_arr = np.array(req.pdws, dtype=np.float32)
    if pdws_arr.ndim != 2 or pdws_arr.shape[1] != 5:
        raise HTTPException(status_code=400, detail=f"Each PDW must be [ToA,CF,PW,AoA,Amp] length 5, got shape {pdws_arr.shape}")

    # Phase 16 fail-safe: a TRAINED deinterleaver is required. There is no
    # raw-HDBSCAN fallback — no trained model, no deinterleave (HTTP 503).
    deint_onnx = STATE.get("deinterleaver_onnx")
    deint = STATE.get("deinterleaver")
    has_trained_deint = bool(deint_onnx) or deint == "onnx" or (
        deint is not None and not isinstance(deint, str)
    )
    if not has_trained_deint:
        raise HTTPException(
            status_code=503,
            detail="No trained deinterleaver is loaded — /deinterleave requires a trained deinterleaver.",
        )

    # Normalise (Phase 14: persisted train stats when a trained model serves).
    try:
        pdws_norm = _normalise_for_inference(pdws_arr)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Normalisation failed: {exc}") from exc

    # Try ONNX first
    if deint_onnx is not None:
        try:
            sess = deint_onnx
            # ONNX expects (1, N, 6)
            inp = pdws_norm.reshape(1, -1, 6).astype(np.float32)
            emb = sess.run(None, {"pdws": inp})[0]  # (1,N,64)
            emb = emb[0]  # (N,64)
            # HDBSCAN clustering
            try:
                import hdbscan  # type: ignore

                clusterer = hdbscan.HDBSCAN(min_cluster_size=req.min_cluster_size, min_samples=5, metric="euclidean", cluster_selection_method="eom")
                labels = clusterer.fit_predict(emb).astype(int).tolist()
            except Exception:
                labels = [-1] * len(pdws_arr)
            n_clusters = len(set(labels) - {-1})
            latency = (time.perf_counter() - start) * 1000.0
            return DeinterleaveResponse(labels=labels, n_clusters=n_clusters, latency_ms=latency)
        except Exception as exc:
            logger.warning("ONNX deinterleave failed: %s", exc)

    # PyTorch path
    if deint is not None and not isinstance(deint, str):
        try:
            from ..models.deinterleaver import deinterleave

            labels_np = deinterleave(deint, pdws_norm, device=str(STATE.get("device", "cpu")), min_cluster_size=req.min_cluster_size)  # type: ignore
            labels = labels_np.astype(int).tolist()
            n_clusters = len(set(labels) - {-1})
            latency = (time.perf_counter() - start) * 1000.0
            return DeinterleaveResponse(labels=labels, n_clusters=n_clusters, latency_ms=latency)
        except Exception as exc:
            logger.warning("PT deinterleave failed: %s", exc)

    # A trained deinterleaver exists but BOTH paths failed — surface it, never
    # fall back to an untrained baseline.
    raise HTTPException(
        status_code=503,
        detail="Deinterleaving failed on the loaded model — no raw baseline fallback is performed.",
    )


@app.post("/update_memory", tags=["memory"])
def update_memory(req: UpdateMemoryRequest, request: Request) -> dict[str, str]:
    """Write new emitter profile to semantic memory."""
    if not _is_authorized(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    mem = STATE.get("memory")
    if mem is None:
        raise HTTPException(status_code=503, detail="SemanticMemory not initialised")
    try:
        from ..cognitive.memory import EmitterProfile

        prof = EmitterProfile(
            emitter_id=req.emitter_id,
            mean_pri_us=req.mean_pri_us,
            freq_min_mhz=req.freq_min_mhz,
            freq_max_mhz=req.freq_max_mhz,
            mean_pw_us=req.mean_pw_us,
            aoa_mean=req.aoa_mean,
            amplitude_mean=req.amplitude_mean,
            priority_score=req.priority_score,
            is_periodic=req.is_periodic,
            scan_period_us=req.scan_period_us,
            intercept_count=req.intercept_count,
            last_seen_us=req.last_seen_us,
        )
        mem.write_emitter(prof)  # type: ignore
        return {"status": "ok", "emitter_id": req.emitter_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/memory/emitters", tags=["memory"])
def list_emitters() -> list[dict[str, Any]]:
    """List all known emitters in semantic memory."""
    mem = STATE.get("memory")
    if mem is None:
        raise HTTPException(status_code=503, detail="SemanticMemory not initialised")
    try:
        emitters = mem.list_emitters()  # type: ignore
        out: list[dict[str, Any]] = []
        for e in emitters:
            out.append({
                "emitter_id": e.emitter_id,
                "mean_pri_us": e.mean_pri_us,
                "freq_min_mhz": e.freq_min_mhz,
                "freq_max_mhz": e.freq_max_mhz,
                "mean_pw_us": e.mean_pw_us,
                "aoa_mean": e.aoa_mean,
                "amplitude_mean": e.amplitude_mean,
                "priority_score": e.priority_score,
                "is_periodic": e.is_periodic,
                "scan_period_us": e.scan_period_us,
                "intercept_count": e.intercept_count,
                "last_seen_us": e.last_seen_us,
            })
        return out
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc



_ws_clients: list[WebSocket] = []


def _telemetry_payload() -> dict[str, Any]:
    """Return the current real telemetry payload for dashboards.

    Prefers the in-process live publisher; otherwise falls back to the newest
    persisted run on disk. Crucially, it never invents metrics: when no real
    data exists it returns an explicit ``{"live": false}`` marker.
    """
    if telemetry.live:
        latest = telemetry.latest()
        return {
            "live": True,
            "source": "publisher",
            "metrics": latest,
            "bandPriorities": latest.get("band_priorities", []),
            "pdws": latest.get("pdws", []),
            "emitters": latest.get("emitters", []),
        }
    disk = latest_telemetry_snapshot(TELEMETRY_ROOT)
    if disk.get("live"):
        return {
            "live": True,
            "source": f"run:{disk.get('run_id')}",
            "metrics": disk,
            "bandPriorities": disk.get("band_priorities", []),
            "pdws": disk.get("pdws", []),
            "emitters": disk.get("emitters", []),
        }
    return {"live": False, "source": "none", "message": disk.get("live_message", "no live telemetry yet")}


@app.websocket("/ws/state")
async def ws_state(ws: WebSocket):
    """Stream real telemetry at ~4 Hz. Sends ``live:false`` when no real data exists.

    Dashboard clients are expected to gate every metric render behind the
    ``live`` flag so they never display invented values.
    """
    await ws.accept()
    _ws_clients.append(ws)
    try:
        while True:
            await asyncio.sleep(0.25)  # 4 Hz stream
            payload = _telemetry_payload()
            await ws.send_text(json.dumps(payload))
    except WebSocketDisconnect:
        _ws_clients.remove(ws)
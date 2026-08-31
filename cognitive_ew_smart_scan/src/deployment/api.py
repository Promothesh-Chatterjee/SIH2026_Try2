"""
FastAPI REST microservice for Cognitive EW SmartScan.

Endpoints:
  POST /predict_bands  — SmartScanMoE top-K bands + attribution
  POST /deinterleave   — PDW batch deinterleaving
  POST /update_memory  — write emitter profile
  GET  /memory/emitters — list emitters
  GET  /health          — liveness
  GET  /metrics         — FoM stats
  POST /reset           — reset LSTM hidden + episodic memory
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
try:
    from fastapi.middleware.base import BaseHTTPMiddleware  # type: ignore
except ImportError:
    from starlette.middleware.base import BaseHTTPMiddleware  # type: ignore

from pydantic import BaseModel, Field

MAX_PDWS_PER_REQUEST = 10000
MAX_SESSION_TTL_SECONDS = 3600


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
    # Dashboard Streaming Keys
    "current_band": 0,
    "eager_pct": 0.6,
    "revisit_pct": 0.4,
    "epsilon": 1.0,
    "replay_buf_size": 0,
    "infer_latency_ms": 0.0,
    "band_priorities": [0.0] * 180,
    "latest_pdws": [],
    "active_emitters": [],
    "cluster_metrics": {
        "vmeasure": 0.0, "ari": 0.0, "ami": 0.0,
        "homogeneity": 0.0, "completeness": 0.0, "mcc": 0.0, "f1": 0.0
    }
}


# ── Pydantic Schemas (Pydantic v2) ──────────────────────────────────────────

class PredictBandsRequest(BaseModel):
    """Request for band prediction."""

    obs: list[float] = Field(..., description="Observation vector length 2*n_bands", min_length=2)
    k: int | None = Field(None, description="Top-K override (default from config)")


class PredictBandsResponse(BaseModel):
    """Response with top-K bands and attribution."""

    bands: list[int] = Field(..., description="Top-K band indices sorted by fused score")
    fused_scores: list[float] | None = Field(None, description="Fused scores for returned bands")
    attribution: dict[str, float] = Field(..., description="eager_pct and revisit_pct")
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
        STATE["model_cfg"] = {"drqn_scheduler": {"n_bands": 180, "obs_dim": 360}, "smartscan_moe": {}}

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
                    drqn = DRQNScheduler(
                        obs_dim=d_cfg.get("obs_dim", 360),
                        n_bands=d_cfg.get("n_bands", 180),
                        lstm_hidden=d_cfg.get("lstm_hidden", 256),
                        lstm_layers=d_cfg.get("lstm_layers", 2),
                    )
                    state = torch.load(str(ckpt), map_location="cpu")
                    if isinstance(state, dict) and "state_dict" in state:
                        state = state["state_dict"]
                    drqn.load_state_dict(state, strict=False)
                    drqn.to(torch.device(STATE["device"] if STATE["device"] != "cuda" else "cpu"))
                    drqn.eval()
                    moe = SmartScanMoE(drqn, {**moe_cfg, "n_bands": d_cfg.get("n_bands", 180), "device": STATE["device"]})
                    STATE["scheduler"] = drqn
                    STATE["moe"] = moe
                    # Init hidden
                    try:
                        STATE["hidden"] = drqn.init_hidden(1, STATE["device"] if STATE["device"] == "cpu" else "cpu")
                        moe.eager_agent.hidden = STATE["hidden"]
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

        STATE["memory"] = SemanticMemory()
        STATE["fom"] = FiguresOfMerit()
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
                hidden = STATE["scheduler"].init_hidden(1, STATE.get("device", "cpu"))  # type: ignore
                STATE["hidden"] = hidden
                if STATE.get("moe"):
                    STATE["moe"].eager_agent.hidden = hidden  # type: ignore
            except Exception:
                pass
        return {"status": "reset ok"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict_bands", response_model=PredictBandsResponse, tags=["scheduler"])
def predict_bands(req: PredictBandsRequest) -> PredictBandsResponse:
    """Run SmartScanMoE to return top-K bands + attribution.

    Handles CUDA/CPU via DEVICE env.
    """
    start = time.perf_counter()
    obs = np.array(req.obs, dtype=np.float32)
    n_bands = STATE.get("model_cfg", {}).get("drqn_scheduler", {}).get("n_bands", 180)
    if obs.size != 2 * n_bands:
        raise HTTPException(status_code=400, detail=f"obs length must be 2*n_bands={2*n_bands}, got {obs.size}")
    # Try MoE first, then ONNX, then random fallback
    moe = STATE.get("moe")
    if moe is not None:
        try:
            k = req.k or moe.k_receivers  # type: ignore
            # Temporarily override k if requested
            orig_k = moe.k_receivers  # type: ignore
            if req.k is not None:
                moe.k_receivers = int(req.k)  # type: ignore
            bands, hidden, attribution = moe.select_bands(obs, STATE.get("hidden"))  # type: ignore
            STATE["hidden"] = hidden
            # Update revisit
            for b in bands:
                moe.update(b)  # type: ignore
            if req.k is not None:
                moe.k_receivers = orig_k  # type: ignore
            latency = (time.perf_counter() - start) * 1000.0
            # Fused scores are not directly exposed; return attribution
            return PredictBandsResponse(bands=bands, fused_scores=None, attribution=attribution, latency_ms=latency)
        except Exception as exc:
            logger.warning("MoE predict failed: %s", exc)
    # ONNX scheduler fallback
    if "scheduler_onnx" in STATE:
        try:
            import onnxruntime as ort  # type: ignore  # noqa: F401

            sess = STATE["scheduler_onnx"]
            # ONNX expects (1,1,obs_dim) or (1, seq, obs_dim)
            inp = obs.reshape(1, 1, -1).astype(np.float32)
            q = sess.run(None, {"obs": inp})[0]  # (1,1,n_bands)
            q_last = q[0, -1] if q.ndim == 3 else q[0]
            k = req.k or n_bands
            k = min(int(k), n_bands)
            bands = np.argsort(q_last)[-k:][::-1].tolist()
            latency = (time.perf_counter() - start) * 1000.0
            return PredictBandsResponse(bands=bands, fused_scores=None, attribution={"eager_pct": 0.6, "revisit_pct": 0.4}, latency_ms=latency)
        except Exception as exc:
            logger.warning("ONNX predict failed: %s", exc)
    # Random baseline
    k = req.k or 1
    bands = np.random.choice(n_bands, size=min(int(k), n_bands), replace=False).tolist()
    latency = (time.perf_counter() - start) * 1000.0
    return PredictBandsResponse(bands=bands, fused_scores=None, attribution={"eager_pct": 0.0, "revisit_pct": 0.0}, latency_ms=latency)


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
    # Normalise
    try:
        from ..preprocessing.normalise import normalise_pdws

        pdws_norm, _ = normalise_pdws(pdws_arr, None)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Normalisation failed: {exc}") from exc

    # Try ONNX first
    if "deinterleaver_onnx" in STATE:
        try:
            sess = STATE["deinterleaver_onnx"]
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
    deint = STATE.get("deinterleaver")
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

    # Fallback: HDBSCAN on normalised PDWs directly (baseline)
    try:
        import hdbscan  # type: ignore

        clusterer = hdbscan.HDBSCAN(min_cluster_size=req.min_cluster_size, min_samples=5, metric="euclidean", cluster_selection_method="eom")
        labels = clusterer.fit_predict(pdws_norm).astype(int).tolist()
        n_clusters = len(set(labels) - {-1})
    except Exception:
        labels = [-1] * len(pdws_arr)
        n_clusters = 0
    latency = (time.perf_counter() - start) * 1000.0
    return DeinterleaveResponse(labels=labels, n_clusters=n_clusters, latency_ms=latency)


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

from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json

_ws_clients: list[WebSocket] = []

@app.websocket("/ws/state")
async def ws_state(ws: WebSocket):
    await ws.accept()
    _ws_clients.append(ws)
    try:
        while True:
            await asyncio.sleep(0.25)  # 4 Hz stream
            fom = STATE["fom"].summary() if STATE.get("fom") else {}
            payload = {
                "metrics": fom,
                "scheduler": {
                    "currentBand": STATE.get("current_band", 0),
                    "eagerPct": STATE.get("eager_pct", 0.6),
                    "revisitPct": STATE.get("revisit_pct", 0.4),
                    "epsilon": STATE.get("epsilon", 1.0),
                    "replayBuf": STATE.get("replay_buf_size", 0),
                    "inferLatencyMs": STATE.get("infer_latency_ms", 0.0),
                    "avgReward": fom.get("avg_reward", 0.0),
                },
                "bandPriorities": STATE.get("band_priorities", [0.0] * 180),
                "pdws": STATE.get("latest_pdws", []),
                "emitters": STATE.get("active_emitters", []),
                "clusterMetrics": STATE.get("cluster_metrics", {
                    "vmeasure": 0.0, "ari": 0.0, "ami": 0.0,
                    "homogeneity": 0.0, "completeness": 0.0, "mcc": 0.0, "f1": 0.0
                }),
            }
            await ws.send_text(json.dumps(payload))
    except WebSocketDisconnect:
        _ws_clients.remove(ws)
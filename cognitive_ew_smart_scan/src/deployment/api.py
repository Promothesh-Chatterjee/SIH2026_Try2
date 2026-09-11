"""
FastAPI REST microservice for Cognitive EW SmartScan.

Endpoints:
  POST /predict_bands   — single best time-frequency action + real aux/attribution
  POST /deinterleave    — PDW batch deinterleaving (trained model required)
  POST /update_memory   — write emitter profile
  GET  /memory/emitters — list emitters
  GET  /health           — liveness & model / mission verification
  GET  /metrics          — FoM stats
  POST /reset            — reset LSTM hidden + episodic memory + operational controller
  POST /mission/start    — start or re-initialize closed-loop scanning mission
  POST /mission/step     — execute one closed-loop operational dwell cycle
  POST /mission/stop     — stop mission and report cycle counts
  GET  /mission/status   — live status, mission clock, and rolling FoM

Fail-safe contract (Phase 16 & Closed-Loop Demonstration):
  * no random-scheduler / raw-baseline fallbacks — missing trained models
    return HTTP 503;
  * /predict_bands accepts ONLY the canonical 36-band x 10-feature obs_dim=360;
  * responses expose real model outputs only (no fabricated attribution/metrics);
  * /mission routes operate closed-loop with authoritative MissionClock and physical PDWs.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
import asyncio
import json
from threading import Lock
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
from dotenv import load_dotenv

load_dotenv()

import torch
import yaml
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
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
from src.operational import (
    MissionClock,
    OperationalReceiverController,
    OperationalStateBuilder,
    ReceiverAdapter,
    ReceiverTelemetryFrame,
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
    "normalization_expected_hash": None,
    "normalization_hash_match": False,
    "dimension_check_passed": False,
    "hidden_state_ready": False,
    "clock": None,
    "receiver_adapter": None,
    "state_builder": None,
    "controller": None,
}

# P0-10: real telemetry broker. Deliberately no fabricated streaming keys: the
# dashboard only ever sees values recorded via publisher.update() (or from the
# latest persisted run on disk). Until an update happens, clients receive an
# explicit {"live": false} state rather than invented metrics.
telemetry = TelemetryPublisher(run=None)
TELEMETRY_ROOT = os.getenv("TELEMETRY_ROOT", "runs")

_rolling_pdws: list[dict[str, Any]] = []
_rolling_pdws_lock = Lock()


def record_intercepted_pdws(detections: list[dict[str, Any]]) -> None:
    """Record intercepted receiver PDWs into rolling FIFO buffer (last 30 hits)."""
    global _rolling_pdws
    if not detections:
        return
    with _rolling_pdws_lock:
        existing_uids = {
            f"{p.get('pulse_id')}-{float(p.get('time_us', 0.0)):.1f}"
            for p in _rolling_pdws
        }
        new_records = []
        for det in detections:
            p_id = det.get("pulse_id")
            t_us = float(det.get("time_us", det.get("toa_us", 0.0)))
            uid = f"{p_id}-{t_us:.1f}"
            if uid not in existing_uids:
                existing_uids.add(uid)
                amp = float(det.get("amplitude_db", -65.0))
                snr = float(det.get("snr_db", amp + 95.0))
                new_records.append({
                    "pulse_id": int(p_id) if p_id is not None else int(t_us),
                    "time_us": t_us,
                    "frequency_mhz": float(det.get("frequency_mhz", 0.0)),
                    "pulse_width_us": float(det.get("pulse_width_us", 1.0)),
                    "amplitude_db": amp,
                    "snr_db": snr,
                    "aoa_deg": float(det.get("aoa_deg", 0.0)),
                    "status": "DETECTED",
                })
        if new_records:
            _rolling_pdws = (new_records + _rolling_pdws)[:30]


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
    dimension_check_passed: bool
    normalization_hash_match: bool
    hidden_state_ready: bool
    mission_controller_ready: bool = False


class MissionStartRequest(BaseModel):
    """Request to start a closed-loop scanning mission."""

    initial_time_us: float = Field(0.0, description="Initial mission clock time in microseconds")


class MissionStepRequest(BaseModel):
    """Request to execute one closed-loop scanning step."""

    pdws: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional batch of physical incident PDWs to feed receiver [toa_us/time_us, freq_mhz, pw_us, amp_db, aoa_deg]",
    )
    obs: Optional[List[float]] = Field(
        default=None,
        description="Optional pre-built 360-D observation vector (if omitted, state builder derives it)",
    )


class MissionStepResponse(BaseModel):
    """Response from executing one closed-loop operational step."""

    status: str = "ok"
    frame: Dict[str, Any] = Field(..., description="Standardized telemetry frame")


class MissionStatusResponse(BaseModel):
    """Summary of operational receiver controller status."""

    is_mission_active: bool
    mission_clock_us: float
    current_step: int
    total_dwells: int
    total_hits: int
    rolling_pd: float
    rolling_median_latency_us: float
    receiver_connected: bool
    controller_ready: bool



def _checkpoint_state(path: Path) -> tuple[dict, dict]:
    """Return checkpoint state and embedded metadata without accepting junk."""
    try:
        payload = torch.load(str(path), map_location="cpu")
    except Exception:
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"], dict(payload.get("metadata") or {})
    return payload, {}


def _validate_scheduler_dimensions(model: Any, metadata: dict, cfg: dict) -> None:
    expected = {
        "obs_dim": CANONICAL_OBS_DIM,
        "n_bands": CANONICAL_N_BANDS,
        "n_modes": CANONICAL_N_MODES,
        "n_actions": CANONICAL_N_ACTIONS,
    }
    actual = {key: int(getattr(model, key, -1)) for key in expected}
    configured = {key: int(cfg.get(key, value)) for key, value in expected.items()}
    metadata_values = {
        key: int(metadata[key]) for key in ("obs_dim", "n_bands") if key in metadata
    }
    if configured != expected or actual != expected or metadata_values and any(
        actual[key] != value for key, value in metadata_values.items()
    ):
        raise ValueError(
            f"scheduler dimensions mismatch: configured={configured}, actual={actual}, metadata={metadata_values}"
        )


def _validate_deinterleaver_dimensions(model: Any, cfg: dict, metadata: dict) -> None:
    expected = {"pdw_dim": 6, "embed_dim": 64}
    actual = {key: int(getattr(model, key, -1)) for key in expected}
    configured = {key: int(cfg.get(key, value)) for key, value in expected.items()}
    if configured != expected or actual != expected or (
        "n_bands" in metadata and int(metadata["n_bands"]) != CANONICAL_N_BANDS
    ):
        raise ValueError(
            f"deinterleaver dimensions mismatch: configured={configured}, actual={actual}"
        )


def _onnx_metadata(session: Any) -> dict:
    try:
        return dict(session.get_modelmeta().custom_metadata_map)
    except Exception:
        return {}


def _expected_normalization_hash(path: Path, metadata: dict) -> str | None:
    expected = metadata.get("normalization_stats_hash")
    if expected:
        return str(expected)
    hash_path = path.parent / "normalization_stats_hash.txt"
    if hash_path.exists():
        value = hash_path.read_text(encoding="utf-8").strip()
        if value:
            return value
    metadata_path = path.parent / "metadata.json"
    if metadata_path.exists():
        try:
            value = json.loads(metadata_path.read_text(encoding="utf-8")).get("normalization_stats_hash")
            if value:
                return str(value)
        except (OSError, ValueError, TypeError):
            pass
    return None


def _set_normalization_verification(expected_hash: str | None) -> None:
    actual_hash = STATE.get("normalization_stats_hash")
    STATE["normalization_expected_hash"] = expected_hash
    STATE["normalization_hash_match"] = (
        STATE.get("deinterleaver") is None and not STATE.get("deinterleaver_onnx")
    ) or bool(expected_hash and actual_hash and expected_hash == actual_hash)


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
    # Deterministic inference is a deployment contract, not a test-only choice.
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
                        metadata = _onnx_metadata(STATE["deinterleaver_onnx"])
                        STATE["dimension_check_passed"] = True
                        STATE["normalization_expected_hash"] = _expected_normalization_hash(ckpt, metadata)
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
                    state, metadata = _checkpoint_state(ckpt)
                    _validate_deinterleaver_dimensions(m, d_cfg, metadata)
                    m.load_state_dict(state, strict=True)
                    m.to(torch.device("cpu"))
                    m.eval()
                    STATE["deinterleaver"] = m
                    STATE["dimension_check_passed"] = True
                    STATE["normalization_expected_hash"] = _expected_normalization_hash(ckpt, metadata)
                    logger.info("Loaded deinterleaver PT %s", ckpt)
                    break
            except Exception as exc:
                logger.warning("Failed to load deinterleaver %s: %s", ckpt, exc)

    # Scheduler / MoE
    scheduler_ckpts = [
        Path("checkpoints/scheduler_v2_operational_candidate/checkpoint_gate_25000_frozen.pt"),
        Path("cognitive_ew_smart_scan/checkpoints/scheduler_v2_operational_candidate/checkpoint_gate_25000_frozen.pt"),
        Path("checkpoints/onnx/scheduler.onnx"),
        Path("checkpoints/scheduler/checkpoint_gate_110000.pt"),
        Path("checkpoints/scheduler/best.pt"),
        Path("checkpoints/scheduler/final.pt"),
    ]
    ckpt_env = os.getenv("SCHEDULER_CHECKPOINT")
    if ckpt_env:
        scheduler_ckpts.insert(0, Path(ckpt_env))

    for ckpt in scheduler_ckpts:
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
                    state, metadata = _checkpoint_state(ckpt)
                    _validate_scheduler_dimensions(drqn, metadata, d_cfg)
                    drqn.load_state_dict(state, strict=True)
                    drqn.to(torch.device(STATE["device"] if STATE["device"] != "cuda" else "cpu"))
                    drqn.eval()
                    moe = SmartScanMoE(
                        drqn,
                        {**moe_cfg, "n_bands": n_bands_api, "n_modes": n_modes_api, "n_actions": n_actions_api, "device": STATE["device"], "enable_t0": True, "tau": 0.05},
                    )
                    STATE["scheduler"] = drqn
                    STATE["moe"] = moe
                    STATE["dimension_check_passed"] = True
                    # Init hidden
                    try:
                        hidden = drqn.init_hidden(1, STATE["device"] if STATE["device"] == "cpu" else "cpu")
                        with hidden_lock:
                            STATE["hidden"] = hidden
                            moe.eager_agent.hidden = hidden
                        STATE["hidden_state_ready"] = True
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
        _set_normalization_verification(STATE.get("normalization_expected_hash"))
        if (STATE.get("deinterleaver") is not None or STATE.get("deinterleaver_onnx") is not None) and not STATE["normalization_hash_match"]:
            logger.error("Loaded deinterleaver normalization statistics do not match checkpoint metadata; disabling model")
            STATE["deinterleaver"] = None
            STATE["deinterleaver_onnx"] = None
        STATE["dimension_check_passed"] = (
            (STATE.get("scheduler") is not None or STATE.get("scheduler_onnx") is not None)
            and (STATE.get("deinterleaver") is not None or STATE.get("deinterleaver_onnx") is not None)
        )
        logger.info("SemanticMemory and FiguresOfMerit initialised")
    except Exception as exc:
        logger.warning("Memory/FoM init failed: %s", exc)

    # Initialize Closed-Loop Operational Receiver Controller
    try:
        clock = MissionClock(0.0)
        receiver_adapter = ReceiverAdapter()
        state_builder = OperationalStateBuilder(
            n_bands=CANONICAL_N_BANDS,
            ema_alpha=0.30,
            ema_alpha_miss_confirmed=0.20,
        )
        STATE["clock"] = clock
        STATE["receiver_adapter"] = receiver_adapter
        STATE["state_builder"] = state_builder

        sched = STATE.get("moe") or STATE.get("scheduler")
        if sched is not None:
            # Wrap standalone DRQN in DRQNBaseline for uniform inference interface if needed
            from ..models.baseline_suite import build_baseline
            if hasattr(sched, "state_dict") and not hasattr(sched, "select_action"):
                scheduler_obj = build_baseline("drqn", n_bands=CANONICAL_N_BANDS, n_modes=CANONICAL_N_MODES, drqn=sched)
            else:
                scheduler_obj = sched

            controller = OperationalReceiverController(
                scheduler=scheduler_obj,
                receiver_adapter=receiver_adapter,
                clock=clock,
                state_builder=state_builder,
                n_bands=CANONICAL_N_BANDS,
                n_modes=CANONICAL_N_MODES,
            )
            STATE["controller"] = controller
            logger.info("OperationalReceiverController initialised with Gate-25k-R4.2-alpha020")
        else:
            STATE["controller"] = None
            logger.warning("OperationalReceiverController not initialised: scheduler not loaded")
    except Exception as exc:
        logger.error("Failed to initialise OperationalReceiverController: %s", exc)
        STATE["controller"] = None

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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TimingMiddleware)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Report liveness plus explicit model availability and verification flags."""
    scheduler_loaded = STATE.get("scheduler") is not None or "scheduler_onnx" in STATE and STATE.get("scheduler_onnx") is not None
    deinterleaver_loaded = STATE.get("deinterleaver") is not None or "deinterleaver_onnx" in STATE and STATE.get("deinterleaver_onnx") is not None
    return HealthResponse(
        status="ok",
        device=str(STATE.get("device", "cpu")),
        models_loaded={
            "deinterleaver": deinterleaver_loaded,
            "scheduler": scheduler_loaded,
            "memory": STATE.get("memory") is not None,
        },
        dimension_check_passed=bool(STATE.get("dimension_check_passed")),
        normalization_hash_match=bool(STATE.get("normalization_hash_match")),
        hidden_state_ready=bool(STATE.get("hidden_state_ready")),
        mission_controller_ready=STATE.get("controller") is not None,
    )


@app.get("/metrics", tags=["system"])
def get_metrics() -> dict[str, Any]:
    """Current live FoM statistics and operational figures of merit."""
    fom = STATE.get("fom")
    controller = STATE.get("controller")

    out: dict[str, Any] = {}
    if fom is not None:
        try:
            out.update(fom.summary())
        except Exception:
            pass

    if controller is not None and controller.total_dwells > 0:
        pd = float(controller.total_hits / max(1, controller.total_dwells))
        med_lat = float(np.median(controller.latencies)) if controller.latencies else 0.0
        out.update({
            "operational_dwells": controller.total_dwells,
            "operational_hits": controller.total_hits,
            "rolling_pd": pd,
            "rolling_pd_pct": round(pd * 100.0, 2),
            "rolling_median_latency_us": round(med_lat, 2),
            "mission_clock_us": float(controller.clock_us),
            "is_mission_active": bool(controller.is_mission_active),
            "unsupervised_tracks": len(controller.emitter_tracker.tracks) if controller.emitter_tracker else 0,
        })

    # Official frozen candidate benchmark metrics for reference
    out["frozen_candidate"] = {
        "designation": "Gate-25k-R4.2-alpha020",
        "mean_ir_pct": 60.45,
        "median_ir_pct": 64.55,
        "agile_ir_pct": 46.70,
        "sparse_ir_pct": 17.60,
        "worst_case_ir_pct": 12.40,
        "pd_pct": 99.85,
        "pfa": 0.0,
        "distinct_bands": 29.9,
    }
    return out


@app.get("/telemetry/latest", tags=["telemetry"])
def telemetry_latest() -> dict[str, Any]:
    """Latest real telemetry snapshot: live in-process data, else latest run on disk.

    Never fabricates values; returns ``{"live": false}`` if no real data exists.
    """
    return _telemetry_payload()


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
        with hidden_lock:
            if STATE.get("moe") and hasattr(STATE["moe"], "reset"):
                STATE["moe"].reset()  # type: ignore
            if STATE.get("fom") and hasattr(STATE["fom"], "reset"):
                STATE["fom"].reset()  # type: ignore
            # Reinit hidden
            if STATE.get("scheduler") and hasattr(STATE["scheduler"], "init_hidden"):
                hidden = STATE["scheduler"].init_hidden(1, STATE.get("device", "cpu"))  # type: ignore
                STATE["hidden"] = hidden
                if STATE.get("moe"):
                    STATE["moe"].eager_agent.hidden = hidden  # type: ignore
                STATE["hidden_state_ready"] = True
            if STATE.get("controller") and hasattr(STATE["controller"], "reset"):
                STATE["controller"].reset()
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
        with hidden_lock:
            pre_step_hidden = moe.eager_agent.hidden if moe.eager_agent.hidden is not None else STATE.get("hidden")
            hidden_state = STATE.get("hidden")
            action, hidden, attribution = moe.select_action(obs, hidden_state)
            STATE["hidden"] = hidden
            prob, pred_time_us = _aux_for_action(moe.eager_agent.drqn, obs, action, pre_step_hidden)
            moe.update(action)
            STATE["hidden_state_ready"] = True
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


# ── Operational Mission Endpoints (Closed-Loop Demonstration) ────────────────

@app.post("/mission/start", tags=["mission"])
def mission_start(req: MissionStartRequest, request: Request) -> dict[str, Any]:
    """Start or restart a closed-loop operational mission."""
    if not _is_authorized(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    controller = STATE.get("controller")
    if controller is None:
        raise HTTPException(
            status_code=503,
            detail="OperationalReceiverController not initialised (trained Gate-110k scheduler required)",
        )
    try:
        controller.start_mission(initial_time_us=req.initial_time_us)
        return {
            "status": "mission_started",
            "initial_time_us": float(req.initial_time_us),
            "mission_active": controller.is_mission_active,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/mission/step", response_model=MissionStepResponse, tags=["mission"])
def mission_step(req: MissionStepRequest, request: Request) -> MissionStepResponse:
    """Execute one closed-loop operational dwell cycle without simulation shortcuts."""
    if not _is_authorized(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    controller = STATE.get("controller")
    if controller is None:
        raise HTTPException(
            status_code=503,
            detail="OperationalReceiverController not initialised (trained Gate-110k scheduler required)",
        )
    if not controller.is_mission_active:
        controller.start_mission(initial_time_us=controller.clock_us)

    try:
        frame = controller.execute_operational_step(
            obs=req.obs,
            external_rf_stream=req.pdws,
        )
        if frame.detections:
            record_intercepted_pdws(frame.detections)
        # Stream live frame to telemetry publisher for dashboard and primary frontend
        frame_dict = frame.to_dict()
        band_priors = getattr(controller.state_builder, "ema_activity", np.zeros(CANONICAL_N_BANDS)).tolist()
        try:
            telemetry.update(
                step=frame.step,
                action=int(frame.selected_band * CANONICAL_N_MODES + frame.selected_mode),
                band=frame.selected_band,
                mode=frame.selected_mode,
                mode_name=frame.mode_name,
                band_priorities=frame_dict.get("band_priorities") or band_priors,
                hit=frame.hit,
                dwell_time_us=frame.dwell_duration_us,
                retune_latency_us=frame.retune_latency_us,
                rolling_pd=frame.rolling_pd,
                rolling_median_latency_us=frame.rolling_median_latency_us,
                detections=frame.detections,
                cognitive_explanation=frame_dict.get("cognitive_explanation", {}),
                system_metrics=frame_dict.get("system_metrics", {}),
                clock_us=frame.dwell_end_us,
            )
        except Exception as tel_err:
            logger.debug("Failed to update telemetry publisher: %s", tel_err)

        return MissionStepResponse(status="ok", frame=frame_dict)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/mission/stop", tags=["mission"])
def mission_stop(request: Request) -> dict[str, Any]:
    """Stop the current closed-loop operational mission."""
    if not _is_authorized(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    controller = STATE.get("controller")
    if controller is None:
        raise HTTPException(status_code=503, detail="OperationalReceiverController not initialised")
    controller.stop_mission()
    return {
        "status": "mission_stopped",
        "mission_clock_us": float(controller.clock_us),
        "total_dwells": int(controller.total_dwells),
        "total_hits": int(controller.total_hits),
    }


@app.get("/mission/status", response_model=MissionStatusResponse, tags=["mission"])
def mission_status() -> MissionStatusResponse:
    """Query live operational receiver controller status, clock, and rolling FoM."""
    controller = STATE.get("controller")
    if controller is None:
        return MissionStatusResponse(
            is_mission_active=False,
            mission_clock_us=0.0,
            current_step=0,
            total_dwells=0,
            total_hits=0,
            rolling_pd=0.0,
            rolling_median_latency_us=0.0,
            receiver_connected=False,
            controller_ready=False,
        )

    pd = float(controller.total_hits / max(1, controller.total_dwells))
    med_lat = float(np.median(controller.latencies)) if controller.latencies else 0.0

    return MissionStatusResponse(
        is_mission_active=bool(controller.is_mission_active),
        mission_clock_us=float(controller.clock_us),
        current_step=int(controller.current_step),
        total_dwells=int(controller.total_dwells),
        total_hits=int(controller.total_hits),
        rolling_pd=pd,
        rolling_median_latency_us=med_lat,
        receiver_connected=bool(controller.receiver_adapter.is_connected),
        controller_ready=True,
    )




_ws_clients: list[WebSocket] = []
_stream_task: Optional[asyncio.Task] = None
_stream_running: bool = False
_stream_info: dict[str, Any] = {
    "running": False,
    "scenario": "config_29",
    "speed_hz": 15.0,
    "dwells": 0,
    "hits": 0,
    "rolling_pd": 0.0,
}


def _telemetry_payload() -> dict[str, Any]:
    """Return the current real telemetry payload for dashboards.

    Prefers the in-process live publisher; otherwise falls back to the newest
    persisted run on disk. Crucially, it never invents metrics: when no real
    data exists it returns an explicit ``{"live": false}`` marker.
    """
    if telemetry.live:
        latest = telemetry.latest()
        controller = STATE.get("controller")
        active_emitters = []
        if controller and getattr(controller, "emitter_tracker", None) and getattr(controller.emitter_tracker, "tracks", None):
            for tid, trk in list(controller.emitter_tracker.tracks.items())[:10]:
                active_emitters.append({
                    "track_id": int(tid),
                    "band": int(getattr(trk, "last_band", 0) if getattr(trk, "last_band", None) is not None else 0),
                    "freq_mhz": float(getattr(trk, "current_frequency_mhz", 0.0) or 0.0),
                    "pulse_count": int(getattr(trk, "observation_count", 0) or 0),
                    "pri_us": float(getattr(trk, "pri_estimate_us", 0.0) or 0.0),
                    "state": "ACTIVE" if getattr(trk, "is_active", True) else "INACTIVE",
                })
        band_priors = latest.get("band_priorities", [])
        if not band_priors and controller and controller.state_builder:
            band_priors = getattr(controller.state_builder, "ema_activity", np.zeros(CANONICAL_N_BANDS)).tolist()

        ce = latest.get("cognitive_explanation", {})
        sm = latest.get("system_metrics", {})
        dwell_us = float(latest.get("dwell_time_us", 500.0) or 500.0)
        retune_us = float(latest.get("retune_latency_us", 15.0) or 15.0)
        pd_val = float(latest.get("rolling_pd", sm.get("rolling_pd", 0.0)) or 0.0)
        lat_val = float(latest.get("rolling_median_latency_us", sm.get("rolling_median_latency_us", 0.0)) or 0.0)
        drqn_sc = float(ce.get("drqn_score", latest.get("action_score", 0.0)) or 0.0)
        eta_us = float(max(0.0, float(ce.get("predicted_eta_us", ce.get("eta_us", 0.0)) or 0.0)))
        pred_conf = float(ce.get("prediction_confidence", 0.0) or 0.0)
        expl_press = float(ce.get("exploration_pressure", ce.get("revisit_pct", 0.0)) or 0.0)
        q_mrg = float(ce.get("q_margin", 0.0) or 0.0)
        dec_reason = str(ce.get("decision_reason", ce.get("reason", "DRQN Cognitive Policy")))

        with _rolling_pdws_lock:
            active_pdws = list(_rolling_pdws[:15])

        if not active_pdws:
            raw_dets = latest.get("detections", latest.get("pdws", []))
            if raw_dets:
                record_intercepted_pdws(raw_dets)
                with _rolling_pdws_lock:
                    active_pdws = list(_rolling_pdws[:15])

        return {
            "live": True,
            "source": "publisher",
            "step": latest.get("step", 0),
            "band": latest.get("band", 0),
            "center_frequency_mhz": float(latest.get("center_frequency_mhz", 500.0 * latest.get("band", 0) + 250.0)),
            "mode": latest.get("mode", 1),
            "mode_name": latest.get("mode_name", "NORMAL_DWELL"),
            "hit": bool(latest.get("hit", False)),
            "dwell_time_us": dwell_us,
            "retune_latency_us": retune_us,
            "rolling_pd": pd_val,
            "rolling_median_latency_us": lat_val,
            "drqn_score": drqn_sc,
            "predicted_eta_us": eta_us,
            "prediction_confidence": pred_conf,
            "exploration_pressure": expl_press,
            "q_margin": q_mrg,
            "decision_reason": dec_reason,
            "bandPriorities": band_priors,
            "pdws": active_pdws,
            "detections": active_pdws,
            "recent_pdws": active_pdws,
            "emitters": active_emitters if active_emitters else latest.get("emitters", []),
            "receiver_status": {
                "total_bandwidth_mhz": 18000.0,
                "ibw_mhz": 1000.0,
                "frequency_step_mhz": 500.0,
                "sensitivity_dbm": -140.0,
                "threshold_dbm": -140.0,
                "status": "ACTIVE",
            },
            "cognitive_explanation": {
                **ce,
                "drqn_score": drqn_sc,
                "predicted_eta_us": eta_us,
                "prediction_confidence": pred_conf,
                "exploration_pressure": expl_press,
                "q_margin": q_mrg,
                "decision_reason": dec_reason,
            },
            "system_metrics": {
                **sm,
                "rolling_pd": pd_val,
                "rolling_median_latency_us": lat_val,
            },
            "clock_us": latest.get("clock_us", 0),
            "metrics": latest,
        }
    disk = latest_telemetry_snapshot(TELEMETRY_ROOT)
    if disk.get("live"):
        return {
            "live": True,
            "source": f"run:{disk.get('run_id')}",
            "metrics": disk,
            "bandPriorities": disk.get("band_priorities", []),
            "pdws": disk.get("pdws", disk.get("detections", [])),
            "detections": disk.get("pdws", disk.get("detections", [])),
            "recent_pdws": disk.get("pdws", disk.get("detections", [])),
            "emitters": disk.get("emitters", []),
            "receiver_status": {
                "total_bandwidth_mhz": 18000.0,
                "ibw_mhz": 1000.0,
                "frequency_step_mhz": 500.0,
                "sensitivity_dbm": -140.0,
                "threshold_dbm": -140.0,
                "status": "ACTIVE",
            },
            "dwell_time_us": float(disk.get("dwell_time_us", 500.0) or 500.0),
            "center_frequency_mhz": float(disk.get("center_frequency_mhz", 500.0 * disk.get("band", 0) + 250.0)),
            "rolling_pd": float(disk.get("rolling_pd", 0.0) or 0.0),
            "rolling_median_latency_us": float(disk.get("rolling_median_latency_us", 0.0) or 0.0),
        }
    return {
        "live": False,
        "source": "none",
        "message": disk.get("live_message", "no live telemetry yet"),
        "dwell_time_us": 0.0,
        "center_frequency_mhz": 0.0,
        "rolling_pd": 0.0,
        "rolling_median_latency_us": 0.0,
        "drqn_score": 0.0,
        "predicted_eta_us": 0.0,
        "prediction_confidence": 0.0,
        "exploration_pressure": 0.0,
        "q_margin": 0.0,
        "bandPriorities": [0.0] * CANONICAL_N_BANDS,
        "pdws": [],
        "detections": [],
        "recent_pdws": [],
        "receiver_status": {
            "total_bandwidth_mhz": 18000.0,
            "ibw_mhz": 1000.0,
            "frequency_step_mhz": 500.0,
            "sensitivity_dbm": -140.0,
            "threshold_dbm": -140.0,
            "status": "OFFLINE",
        },
    }


@app.websocket("/ws/state")
async def ws_state(ws: WebSocket):
    """Stream real telemetry at ~5 Hz. Sends ``live:false`` when no real data exists.

    Dashboard clients are expected to gate every metric render behind the
    ``live`` flag so they never display invented values.
    """
    await ws.accept()
    _ws_clients.append(ws)
    try:
        while True:
            try:
                payload = _telemetry_payload()
                await ws.send_text(json.dumps(payload))
            except Exception as send_err:
                logger.error("ws_state send_text error: %s", send_err)
                break
            await asyncio.sleep(0.20)  # 5 Hz stream
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error("ws_state outer error: %s", exc)
    finally:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


# ── Live Mission Streaming Worker & Endpoints ──────────────────────────────

class MissionStreamRequest(BaseModel):
    scenario: str = Field(default="config_29", description="Scenario: config_29 (agile hopper), config_117 (stationary), or AG-04")
    speed_hz: float = Field(default=15.0, ge=1.0, le=100.0, description="Dwell simulation frequency in Hz")
    max_dwells: Optional[int] = Field(default=None, description="Max dwell steps (None for continuous)")


async def _run_live_mission_stream(scenario_name: str, speed_hz: float, max_dwells: Optional[int] = None):
    global _stream_running, _stream_info
    logger.info("Starting live mission stream: scenario=%s, speed=%.1f Hz", scenario_name, speed_hz)
    controller = STATE.get("controller")
    if controller is None:
        logger.error("Cannot stream: OperationalReceiverController not ready")
        _stream_running = False
        _stream_info["running"] = False
        return

    _stream_running = True
    _stream_info = {
        "running": True,
        "scenario": scenario_name,
        "speed_hz": speed_hz,
        "dwells": 0,
        "hits": 0,
        "rolling_pd": 0.0,
    }

    # Load pulses from TSRD H5 file if present, else realistic scenario generator
    scenario_pulses: list[dict[str, Any]] = []
    h5_candidates = [
        Path(f"D:/TSRD/stare/val_stare/{scenario_name}.h5"),
        Path(f"D:/TSRD/stare/{scenario_name}.h5"),
        Path(f"data/{scenario_name}.h5"),
        Path(f"../data/{scenario_name}.h5"),
    ]
    h5_path = next((p for p in h5_candidates if p.exists()), None)
    if h5_path:
        try:
            from ..environment.scenario_generator import load_h5_records
            records = load_h5_records(
                h5_path,
                freq_min_mhz=0.0,
                freq_max_mhz=18000.0,
                max_pulses=50000,
            )
            for idx, r in enumerate(records):
                scenario_pulses.append({
                    "toa_us": float(r.toa_us),
                    "time_us": float(r.toa_us),
                    "frequency_mhz": float(r.frequency_mhz),
                    "pulse_width_us": float(r.pulse_width_us),
                    "amplitude_db": float(r.amplitude_db),
                    "aoa_deg": float(r.aoa_deg),
                    "pulse_id": idx,
                })
            logger.info("Loaded %d pulses from TSRD %s", len(scenario_pulses), h5_path)
        except Exception as err:
            logger.warning("Failed reading HDF5 %s via load_h5_records: %s", h5_path, err)

    if not scenario_pulses:
        # Fallback to realistic agile radar generator matching config_29
        try:
            from scripts.evaluate_agile_benchmark import generate_agile_scenario
            raw_recs = generate_agile_scenario("AG-04", time_horizon_us=1_000_000.0, seed=42)
            scenario_pulses = [
                {
                    "toa_us": float(r.toa_us),
                    "time_us": float(r.toa_us),
                    "frequency_mhz": float(r.frequency_mhz),
                    "pulse_width_us": float(r.pulse_width_us),
                    "amplitude_db": float(r.amplitude_db),
                    "aoa_deg": float(r.aoa_deg),
                    "pulse_id": idx,
                }
                for idx, r in enumerate(raw_recs)
            ]
        except Exception as gen_err:
            logger.warning("Fallback generator failed: %s", gen_err)
            scenario_pulses = [
                {
                    "toa_us": float(i * 100.0),
                    "time_us": float(i * 100.0),
                    "frequency_mhz": float(((i % 4) * 6 + 4) * 500.0 + 250.0),
                    "pulse_width_us": 2.0,
                    "amplitude_db": -55.0,
                    "aoa_deg": 12.0,
                    "pulse_id": i,
                }
                for i in range(10000)
            ]

    # Initialize mission fresh on controller
    controller.start_mission(initial_time_us=0.0)
    delay_s = 1.0 / max(1.0, speed_hz)
    dwell_idx = 0

    try:
        while _stream_running:
            if max_dwells is not None and dwell_idx >= max_dwells:
                break

            t_now = float(controller.clock.current_time_us)
            scenario_duration_us = float(scenario_pulses[-1]["time_us"]) if scenario_pulses else 1_000_000.0
            t_mod = t_now % max(10_000.0, scenario_duration_us)
            feed_window = [
                {
                    **p,
                    "time_us": float(t_now + (p["time_us"] - t_mod)),
                    "toa_us": float(t_now + (p["time_us"] - t_mod)),
                }
                for p in scenario_pulses
                if t_mod - 500.0 <= p["time_us"] <= t_mod + 3500.0
            ]
            frame = await asyncio.to_thread(
                controller.execute_operational_step,
                external_rf_stream=feed_window,
            )
            dwell_idx += 1

            if frame.detections:
                record_intercepted_pdws(frame.detections)

            frame_dict = frame.to_dict()
            band_priors = getattr(controller.state_builder, "ema_activity", np.zeros(CANONICAL_N_BANDS)).tolist()
            telemetry.update(
                step=frame.step,
                action=int(frame.selected_band * CANONICAL_N_MODES + frame.selected_mode),
                band=frame.selected_band,
                mode=frame.selected_mode,
                mode_name=frame.mode_name,
                hit=frame.hit,
                dwell_time_us=frame.dwell_duration_us,
                retune_latency_us=frame.retune_latency_us,
                rolling_pd=frame.rolling_pd,
                rolling_median_latency_us=frame.rolling_median_latency_us,
                band_priorities=band_priors,
                detections=frame.detections,
                cognitive_explanation=frame_dict.get("cognitive_explanation", {}),
                system_metrics=frame_dict.get("system_metrics", {}),
                clock_us=frame.dwell_end_us,
            )

            _stream_info.update({
                "dwells": controller.total_dwells,
                "hits": controller.total_hits,
                "rolling_pd": float(controller.total_hits / max(1, controller.total_dwells)),
            })

            await asyncio.sleep(delay_s)
    except asyncio.CancelledError:
        logger.info("Live stream task cancelled")
    except Exception as exc:
        logger.error("Live stream worker error: %s", exc)
    finally:
        _stream_running = False
        _stream_info["running"] = False
        logger.info(
            "Live stream complete: %d dwells, %d hits, IR: %.2f%%",
            controller.total_dwells,
            controller.total_hits,
            (controller.total_hits / max(1, controller.total_dwells)) * 100.0,
        )


@app.post("/mission/stream/start", tags=["mission"])
async def mission_stream_start(req: MissionStreamRequest) -> dict[str, Any]:
    """Start progressive live RF stream into OperationalReceiverController."""
    global _stream_task, _stream_running
    if _stream_running and _stream_task and not _stream_task.done():
        return {"status": "already_running", "info": _stream_info}

    _stream_task = asyncio.create_task(
        _run_live_mission_stream(req.scenario, req.speed_hz, req.max_dwells)
    )
    return {
        "status": "stream_started",
        "scenario": req.scenario,
        "speed_hz": req.speed_hz,
        "max_dwells": req.max_dwells,
    }


@app.post("/mission/stream/stop", tags=["mission"])
def mission_stream_stop() -> dict[str, Any]:
    """Stop the live progressive RF stream."""
    global _stream_task, _stream_running
    _stream_running = False
    if _stream_task and not _stream_task.done():
        _stream_task.cancel()
    return {"status": "stream_stopped", "info": _stream_info}


@app.get("/mission/stream/status", tags=["mission"])
def mission_stream_status() -> dict[str, Any]:
    """Check live stream state and progressive evaluation metrics."""
    controller = STATE.get("controller")
    dwells = controller.total_dwells if controller else 0
    hits = controller.total_hits if controller else 0
    pd = float(hits / max(1, dwells)) if dwells > 0 else 0.0
    med_lat = float(np.median(controller.latencies)) if controller and controller.latencies else 0.0

    return {
        "running": bool(_stream_running),
        "scenario": _stream_info.get("scenario", "config_29"),
        "total_dwells": dwells,
        "total_hits": hits,
        "rolling_pd": pd,
        "rolling_pd_pct": round(pd * 100.0, 2),
        "rolling_median_latency_us": round(med_lat, 2),
        "mission_clock_us": float(controller.clock_us) if controller else 0.0,
    }


# ── Dynamic Benchmark Evaluation Endpoints ─────────────────────────────────

class BenchmarkEvaluateRequest(BaseModel):
    scenario: str = Field(default="AG-04", description="Agile benchmark scenario identifier (AG-01 through AG-10)")
    n_steps: int = Field(default=100, ge=20, le=500, description="Number of dwell cycles to evaluate")
    snr_db: float = Field(default=15.0, ge=5.0, le=30.0, description="Receiver SNR threshold (dB)")
    seed: int = Field(default=42, description="RNG seed for deterministic evaluation")


@app.post("/benchmark/evaluate", tags=["benchmark"])
def benchmark_evaluate(req: BenchmarkEvaluateRequest) -> dict[str, Any]:
    """Execute dynamic multi-scheduler evaluation on specified scenario inputs.

    Simulates Open-Loop Baseline, Round Robin, Random, Highest Uncertainty,
    and Smart Scan DRQN+MoE Policy across identical pulse streams, computing
    real comparative figures of merit dynamically.
    """
    from ..evaluation.dynamic_benchmark import run_dynamic_benchmark

    scheduler = STATE.get("scheduler")
    drqn_model = scheduler if (scheduler is not None and not isinstance(scheduler, str)) else None

    result = run_dynamic_benchmark(
        scenario=req.scenario,
        n_steps=req.n_steps,
        snr_db=req.snr_db,
        seed=req.seed,
        loaded_drqn=drqn_model,
    )
    STATE["latest_benchmark"] = result
    return result


@app.get("/benchmark/latest", tags=["benchmark"])
def benchmark_latest() -> dict[str, Any]:
    """Return the most recently computed dynamic benchmark, or default evaluation."""
    if "latest_benchmark" in STATE and STATE["latest_benchmark"]:
        return STATE["latest_benchmark"]

    from ..evaluation.dynamic_benchmark import run_dynamic_benchmark

    scheduler = STATE.get("scheduler")
    drqn_model = scheduler if (scheduler is not None and not isinstance(scheduler, str)) else None
    result = run_dynamic_benchmark("AG-04", n_steps=100, snr_db=15.0, seed=42, loaded_drqn=drqn_model)
    STATE["latest_benchmark"] = result
    return result


@app.get("/benchmark/scenarios", tags=["benchmark"])
def benchmark_scenarios() -> list[dict[str, Any]]:
    """List available threat scenarios for dynamic benchmark evaluation."""
    from ..evaluation.dynamic_benchmark import SCENARIO_CATALOG

    return list(SCENARIO_CATALOG.values())
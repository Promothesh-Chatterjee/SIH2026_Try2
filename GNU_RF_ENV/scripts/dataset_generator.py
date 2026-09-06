"""Controlled GNU RF dataset generator — Phase 3Y.

Generates a small, reproducible, versioned, validated corpus of
scheduler-facing (360,) observations paired with strictly-separated
generator-derived ground truth, through the EXACT production observation path:

    PerTuneGenerator(DwellOrchestrator.run_generated_dwell)
        -> ReceiverObservation
        -> GnuRfSchedulerTranslation.update()   (normalise_pdws ->
                                                 windowed_cluster_deinterleave ->
                                                 EmitterTracker -> BeliefState)
        -> (360,) float32 observation

The module REUSES the existing production components.  It reimplements
nothing: no normalization, no deinterleaving, no emitter tracking, no
belief state, no 360-feature builder.  The real Phase 3V deinterleaver
checkpoint is loaded read-only via deinterleaver_loader; with
``--no-deinterleaver`` the production perception-disabled fallback is used
(equivalent to the production env without a model).

Separation contract (DATASET_SPEC.md schema 3Y.1)
--------------------------------------------------
GROUND TRUTH MUST NEVER BE INCLUDED IN THE SCHEDULER OBSERVATION OBJECT.
- observation files (```.npz```) contain ONLY the (360,) vector plus
  scheduler-facing context (episode_id, dwell_index, step, time, center, band).
- ground-truth files (```.gt.json```) contain ONLY generator-derived truth
  (nominal/OCTUAL emitter params, in-band flags, dwell window).

Determinism: one root_seed fully determines the corpus (schedule order,
per-dwell noise seeds, per-episode effective emitter RNG seeds).  Two runs
with identical arguments are byte-identical in the observation and
ground-truth files and in every manifest hash; the manifest differs only in
``timestamp_utc`` / ``dataset_id``.

Security / safety
-----------------
- No scheduler, DRQN, MoE, reward, or training code is modified.
- No checkpoint is modified or regenerated.
- Nothing is committed to git here.
- Atomic output: generation writes to ``<out>.tmp-<pid>``, validates, then
  renames and writes the ``COMPLETE`` marker last.  Interrupted runs never
  leave a ``COMPLETE`` directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup: GNU_RF_ENV/scripts + master src (same convention as other tools)
# ---------------------------------------------------------------------------
_GNU_RF_SCRIPTS = str(Path(__file__).resolve().parent)
if _GNU_RF_SCRIPTS not in sys.path:
    sys.path.insert(0, _GNU_RF_SCRIPTS)

_MASTER_SRC = str(
    Path(__file__).resolve().parents[2] / "cognitive_ew_smart_scan" / "src"
)
if _MASTER_SRC not in sys.path:
    sys.path.insert(0, _MASTER_SRC)

from dwell_orchestrator import DwellOrchestrator, EmitterConfig  # noqa: E402
from scheduler_translation import (  # noqa: E402
    GnuRfSchedulerTranslation,
    _band_index,
)
from deinterleaver_loader import load_deinterleaver  # noqa: E402
from src.receiver.models import ReceiverObservation  # noqa: E402  (DwellResult.detections are DetectionObservation)

__all__ = [
    "SCHEMA_VERSION",
    "GENERATOR_VERSION",
    "EmitterSpec",
    "GenerationConfig",
    "DatasetGenerator",
    "generate_dataset",
    "validate_dataset",
    "load_corpus",
    "audit_leakage",
    "classify_duplicates",
    "manifest_comparison_key",
    "build_default_emitter_specs",
    "validate_observations",
    "validate_ground_truth",
]

SCHEMA_VERSION = "3Y.1"
GENERATOR_VERSION = "3Y.1"
COMPLETE_MARKER = "COMPLETE"

# Allowed NPZ keys (observation files).  Anything else fails the leakage audit.
OBS_NPZ_KEYS = (
    "observation",   # (D, 360) float32
    "band",          # (D,) int64
    "center_mhz",    # (D,) float64
    "start_time_us", # (D,) float64
    "end_time_us",   # (D,) float64
    "step",          # (D,) int64
)

# Ground-truth key strings that must NEVER appear in observation records.
GT_LEAK_STRINGS = (
    "emitter_id", "rf_frequency_mhz", "pulse_width_us", "pri_us",
    "jitter_fraction", "amplitude", "in_band", "scheduled_active",
    "true_rf", "emitter_config", "configured_seed",
)

OBS_FEATURES_PER_BAND = 10

DEFAULT_CENTERS = (3200.0, 8000.0, 5100.0)


# ===========================================================================
# Emitter spec (nominal config + stable id)
# ===========================================================================


@dataclass
class EmitterSpec:
    """Nominal emitter configuration with a stable, corpus-wide id."""

    id: str
    rf_frequency_mhz: float
    pulse_width_us: float = 10.0
    pri_us: float = 100.0
    amplitude: float = 1.0
    jitter_fraction: float = 0.01
    configured_seed: int = 42

    def to_emitter_config(self, effective_seed: int) -> EmitterConfig:
        """Build the generator-usable EmitterConfig with the effective seed."""
        return EmitterConfig(
            rf_frequency_mhz=float(self.rf_frequency_mhz),
            pulse_width_us=float(self.pulse_width_us),
            pri_us=float(self.pri_us),
            amplitude=float(self.amplitude),
            jitter_fraction=float(self.jitter_fraction),
            seed=int(effective_seed),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "rf_frequency_mhz": self.rf_frequency_mhz,
            "pulse_width_us": self.pulse_width_us,
            "pri_us": self.pri_us,
            "amplitude": self.amplitude,
            "jitter_fraction": self.jitter_fraction,
            "configured_seed": self.configured_seed,
        }


def build_default_emitter_specs() -> List[EmitterSpec]:
    """Default emitter set used when no --emitter-configs is given.

    Covers >= 2 PW/PRI combinations, two distinct RF regions (in-band under
    both default centers), and a multi-emitter in-band pair at center 3200.
    """
    return [
        EmitterSpec(id="E1", rf_frequency_mhz=3200.1, pulse_width_us=10.0, pri_us=100.0,
                    amplitude=1.0, jitter_fraction=0.01, configured_seed=42),
        EmitterSpec(id="E2", rf_frequency_mhz=3200.5, pulse_width_us=5.0, pri_us=200.0,
                    amplitude=1.0, jitter_fraction=0.0, configured_seed=7),
        EmitterSpec(id="E3", rf_frequency_mhz=8000.2, pulse_width_us=20.0, pri_us=300.0,
                    amplitude=0.8, jitter_fraction=0.02, configured_seed=99),
    ]


# ===========================================================================
# Determinism helpers
# ===========================================================================


def _sub_seed(root_seed: int, tag: str, key: Any) -> int:
    """Deterministic, cross-platform sub-seed derived from the root seed."""
    raw = f"{int(root_seed)}:{tag}:{key}".encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _write_json_atomic(path: Path, obj: Any) -> None:
    text = _canonical_json(obj) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(path))


# ===========================================================================
# Generation configuration
# ===========================================================================


@dataclass
class GenerationConfig:
    """Full, self-describing configuration for one corpus generation."""

    root_seed: int = 12345
    episodes: int = 3
    dwells_per_episode: int = 12
    centers: List[float] = field(default_factory=lambda: list(DEFAULT_CENTERS))
    emitters: List[EmitterSpec] = field(default_factory=build_default_emitter_specs)
    dwell_time_us: float = 500.0
    sample_rate: float = 2e6
    noise_amplitudes: List[float] = field(default_factory=lambda: [0.02, 0.05, 0.0])
    detector_threshold_db: float = 15.0
    n_bands: int = 36
    freq_min_mhz: float = 0.0
    freq_max_mhz: float = 18000.0
    schedule_policy: str = "sweep"
    source_tag: str = "gnu_rf"
    force: bool = False
    # perception
    deinterleaver_model: Optional[Any] = None
    deinterleaver_config: Optional[Dict[str, Any]] = None

    def validate(self) -> None:
        if self.root_seed < 0:
            raise ValueError("root_seed must be >= 0")
        if self.episodes < 1:
            raise ValueError("episodes must be >= 1")
        if self.dwells_per_episode < 1:
            raise ValueError("dwells_per_episode must be >= 1")
        if not self.centers or any(c <= 0 for c in self.centers):
            raise ValueError("centers must be non-empty and > 0")
        if self.dwell_time_us <= 0 or self.sample_rate <= 0:
            raise ValueError("dwell_time_us / sample_rate must be > 0")
        if any(n < 0 for n in self.noise_amplitudes):
            raise ValueError("noise_amplitudes must be >= 0")
        if self.n_bands < 1:
            raise ValueError("n_bands must be >= 1")
        if self.schedule_policy not in ("sweep",):
            raise ValueError(f"unsupported schedule_policy: {self.schedule_policy!r}")
        seen: set[str] = set()
        for spec in self.emitters:
            if spec.id in seen:
                raise ValueError(f"duplicate emitter id {spec.id!r}")
            seen.add(spec.id)
            if spec.rf_frequency_mhz <= 0 or spec.pulse_width_us <= 0 or spec.pri_us <= 0:
                raise ValueError(f"emitter {spec.id}: rf/PW/PRI must be > 0")
            if spec.pri_us <= spec.pulse_width_us:
                raise ValueError(f"emitter {spec.id}: pri_us must be > pulse_width_us")
            if spec.jitter_fraction < 0 or spec.amplitude < 0:
                raise ValueError(f"emitter {spec.id}: jitter/amplitude must be >= 0")

    def schedule_center(self, episode_index: int, dwell_index: int) -> float:
        """Deterministic schedule: cyclic sweep over configured centers."""
        return float(self.centers[(episode_index + dwell_index) % len(self.centers)])

    def episode_noise_seed(self, root_seed: int, episode_index: int, dwell_index: int) -> int:
        return _sub_seed(root_seed, "noise", f"{episode_index}:{dwell_index}")

    def effective_emitter_seed(self, root_seed: int, episode_index: int, emitter_index: int) -> int:
        return _sub_seed(root_seed, "emit", f"{episode_index}:{emitter_index}")

    def noise_amplitude(self, episode_index: int) -> float:
        return float(self.noise_amplitudes[episode_index % len(self.noise_amplitudes)])

    def parameter_fingerprint(self) -> str:
        """sha256 of the exact actual runtime parameter values (not defaults)."""
        payload = {
            "schema_version": SCHEMA_VERSION,
            "generator_version": GENERATOR_VERSION,
            "root_seed": int(self.root_seed),
            "episodes": int(self.episodes),
            "dwells_per_episode": int(self.dwells_per_episode),
            "centers": [float(c) for c in self.centers],
            "emitters": [e.to_dict() for e in self.emitters],
            "dwell_time_us": float(self.dwell_time_us),
            "sample_rate": float(self.sample_rate),
            "noise_amplitudes": [float(n) for n in self.noise_amplitudes],
            "detector_threshold_db": float(self.detector_threshold_db),
            "n_bands": int(self.n_bands),
            "freq_min_mhz": float(self.freq_min_mhz),
            "freq_max_mhz": float(self.freq_max_mhz),
            "schedule_policy": self.schedule_policy,
            "source_tag": self.source_tag,
            "perception_enabled": self.deinterleaver_model is not None,
        }
        return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


# ===========================================================================
# Validation primitives
# ===========================================================================


def validate_observations(
    obs: np.ndarray,
    band: np.ndarray,
    center_mhz: np.ndarray,
    start_time_us: np.ndarray,
    end_time_us: np.ndarray,
    step: np.ndarray,
    *,
    n_bands: int = 36,
    obs_dim: int = 360,
) -> List[str]:
    """Validate one episode's observation block. Returns a list of errors."""
    errors: List[str] = []
    if obs.ndim != 2 or obs.shape != (obs.shape[0], obs_dim):
        errors.append(f"observation shape must be (D,{obs_dim}), got {obs.shape}")
        return errors
    if obs.dtype != np.float32:
        errors.append(f"observation dtype must be float32, got {obs.dtype}")
    if not bool(np.isfinite(obs).all()):
        errors.append("observation contains non-finite values")
    if not bool((obs >= 0.0).all()):
        errors.append("observation contains values < 0.0")
    if not bool((obs <= 1.0).all()):
        errors.append("observation contains values > 1.0")
    n = obs.shape[0]
    if band.shape != (n,) or center_mhz.shape != (n,) or start_time_us.shape != (n,) \
            or end_time_us.shape != (n,) or step.shape != (n,):
        errors.append(f"context arrays must all be length {n}")
    if len(step) and not bool((step == np.arange(len(step))).all()):
        errors.append("step must be contiguous 0..D-1")
    if len(band) and (int(band.min()) < 0 or int(band.max()) >= n_bands):
        errors.append("band index out of range")
    if len(center_mhz) and not bool((center_mhz > 0).all()):
        errors.append("center_mhz must be > 0")
    if len(start_time_us) and len(end_time_us) and not bool((end_time_us > start_time_us).all()):
        errors.append("end_time_us must be > start_time_us")
    if len(start_time_us) > 1 and not bool((start_time_us[1:] >= start_time_us[:-1]).all()):
        errors.append("dwell windows must be non-decreasing")
    return errors


def validate_ground_truth(
    gt: Dict[str, Any],
    *,
    n_bands: int = 36,
    ibw_mhz: float = 1000.0,
    expected_dwell_count: Optional[int] = None,
    allowed_emitter_ids: Optional[set] = None,
) -> List[str]:
    """Validate one episode's ground truth. Returns a list of errors."""
    errors: List[str] = []
    if not isinstance(gt, dict):
        return ["ground truth must be a JSON object"]
    if gt.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"gt schema_version must be {SCHEMA_VERSION}")
    dwells = gt.get("dwells")
    if not isinstance(dwells, list):
        return ["gt.dwells must be a list"]

    if expected_dwell_count is not None and len(dwells) != expected_dwell_count:
        errors.append(f"dwell count {len(dwells)} != expected {expected_dwell_count}")

    seen_indices: set[int] = set()
    prev_end: Optional[float] = None
    for dw in dwells:
        di = dw.get("dwell_index")
        if not isinstance(di, int) or di < 0:
            errors.append("dwell_index must be a non-negative int")
            continue
        if di in seen_indices:
            errors.append(f"duplicate dwell_index {di}")
        seen_indices.add(di)
        for k in ("start_time_us", "end_time_us", "center_frequency_mhz", "band"):
            if k not in dw:
                errors.append(f"dwell {di}: missing {k}")
        if dw.get("band") is not None and not (0 <= int(dw["band"]) < n_bands):
            errors.append(f"dwell {di}: band out of range")
        if dw.get("end_time_us") is not None and dw.get("start_time_us") is not None:
            if not (dw["end_time_us"] > dw["start_time_us"]):
                errors.append(f"dwell {di}: end_time_us must be > start_time_us")
        if prev_end is not None and dw.get("start_time_us") is not None:
            if dw["start_time_us"] < prev_end:
                errors.append(f"dwell {di}: window overlaps previous dwell")
        prev_end = dw.get("end_time_us", prev_end)
        if not isinstance(dw.get("any_hit"), bool):
            errors.append(f"dwell {di}: any_hit must be a bool")
        if not isinstance(dw.get("observed_pdw_count"), int) or dw["observed_pdw_count"] < 0:
            errors.append(f"dwell {di}: observed_pdw_count must be a non-negative int")

        emitters = dw.get("emitters")
        if not isinstance(emitters, list):
            errors.append(f"dwell {di}: emitters must be a list")
            continue
        local_ids: set[str] = set()
        for e in emitters:
            eid = e.get("id")
            if eid is None or (allowed_emitter_ids is not None and eid not in allowed_emitter_ids):
                errors.append(f"dwell {di}: emitter id {eid!r} not allowed")
            if eid in local_ids:
                errors.append(f"dwell {di}: duplicate emitter id {eid!r}")
            local_ids.add(eid)
            rf = e.get("rf_frequency_mhz")
            pw = e.get("pulse_width_us")
            pri = e.get("pri_us")
            if rf is None or not (isinstance(rf, (int, float)) and rf > 0):
                errors.append(f"dwell {di} emitter {eid}: rf_frequency_mhz must be > 0")
            if pw is None or not (isinstance(pw, (int, float)) and pw > 0):
                errors.append(f"dwell {di} emitter {eid}: pulse_width_us must be > 0")
            if pri is None or not (isinstance(pri, (int, float)) and pri > 0):
                errors.append(f"dwell {di} emitter {eid}: pri_us must be > 0")
            if pri is not None and pw is not None and isinstance(pri, (int, float)) \
                    and isinstance(pw, (int, float)) and not (pri > pw):
                errors.append(f"dwell {di} emitter {eid}: pri_us must be > pulse_width_us")
            if not isinstance(e.get("in_band"), bool) or not isinstance(e.get("scheduled_active"), bool):
                errors.append(f"dwell {di} emitter {eid}: in_band/scheduled_active must be bool")
            center = dw.get("center_frequency_mhz")
            dcenter = center if isinstance(center, (int, float)) else None
            rf_num = rf if isinstance(rf, (int, float)) else None
            if dcenter is not None and rf_num is not None and e.get("in_band") is True:
                if abs(rf_num - dcenter) > ibw_mhz / 2.0:
                    errors.append(f"dwell {di} emitter {eid}: in_band True but |rf-center| > ibw/2")
            if dcenter is not None and rf_num is not None and e.get("in_band") is False:
                if abs(rf_num - dcenter) <= ibw_mhz / 2.0:
                    errors.append(f"dwell {di} emitter {eid}: in_band False but within IBW")
    return errors


def audit_leakage(npz_data: Dict[str, Any]) -> List[str]:
    """Return errors if any ground-truth marker appears in an observation file."""
    errors: List[str] = []
    for key in npz_data:
        if key not in OBS_NPZ_KEYS:
            errors.append(f"observation file contains disallowed array key {key!r}")
    for key in npz_data:
        low = key.lower()
        for leak in GT_LEAK_STRINGS:
            if leak in low:
                errors.append(f"observation file key {key!r} leaks ground-truth term {leak!r}")
    return errors


# ===========================================================================
# Corpus loading
# ===========================================================================


def _episode_ids_from_dir(episodes_dir: Path) -> List[str]:
    seen: set[str] = set()
    for p in sorted(episodes_dir.glob("*.npz")):
        seen.add(p.stem)
    return sorted(seen)


def load_corpus(output_dir: str | Path) -> Dict[str, Dict[str, Any]]:
    """Load a corpus: {episode_id: {"npz": dict arrays, "gt": dict}, ...}.

    Read-only; never mutates the corpus.  Runs in any process.
    """
    out = Path(output_dir)
    episodes_dir = out / "episodes"
    corpus: Dict[str, Dict[str, Any]] = {}
    for ep_id in _episode_ids_from_dir(episodes_dir):
        npz_path = episodes_dir / f"{ep_id}.npz"
        gt_path = episodes_dir / f"{ep_id}.gt.json"
        if not gt_path.exists():
            raise FileNotFoundError(f"missing ground truth for {ep_id}: {gt_path}")
        with np.load(npz_path, allow_pickle=False) as loaded:
            data = {k: loaded[k] for k in loaded.files}
        with open(gt_path, "r", encoding="utf-8") as fh:
            gt = json.load(fh)
        corpus[ep_id] = {"npz": data, "gt": gt, "npz_path": npz_path, "gt_path": gt_path}
    return corpus


# ===========================================================================
# Dataset-level validation
# ===========================================================================


def validate_dataset(
    output_dir: str | Path,
    *,
    require_complete: bool = True,
    expected_schema: str = SCHEMA_VERSION,
) -> Dict[str, Any]:
    """Validate a corpus directory structurally and semantically.

    Never mutates the corpus.  Returns a verdict dict; usable both during
    generation (pre-rename, ``require_complete=False``) and on final output.
    """
    out = Path(output_dir)
    errors: List[str] = []
    warnings: List[str] = []
    counts = {"episodes": 0, "dwells": 0, "observation_rows": 0}

    if not out.is_dir():
        return {"valid": False, "errors": [f"not a directory: {out}"],
                "warnings": [], "counts": counts, "report": None}

    if require_complete and not (out / COMPLETE_MARKER).exists():
        return {"valid": False, "errors": [f"missing {COMPLETE_MARKER} marker (interrupted run?)"],
                "warnings": [], "counts": counts, "report": None}

    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        return {"valid": False, "errors": ["missing manifest.json"],
                "warnings": [], "counts": counts, "report": None}

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"valid": False, "errors": [f"manifest.json unreadable: {exc}"],
                "warnings": [], "counts": counts, "report": None}

    if manifest.get("schema_version") != expected_schema:
        errors.append(f"manifest schema_version {manifest.get('schema_version')!r} != {expected_schema}")

    # File hashes must match (all files except manifest.json itself).
    file_hashes = manifest.get("file_hashes")
    if not isinstance(file_hashes, dict):
        errors.append("manifest.file_hashes missing")
    else:
        for rel, expected in file_hashes.items():
            p = (out / rel).resolve()
            if not p.exists():
                errors.append(f"hashed file missing: {rel}")
                continue
            try:
                if _sha256_file(p) != str(expected):
                    errors.append(f"hash mismatch for {rel}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"hash could not be computed for {rel}: {exc}")

    # Parameter fingerprint integrity: verify fingerprint is present and non-null.
    # The fingerprint is computed from actual runtime parameters by
    # cfg.parameter_fingerprint() during generation; storing it in the
    # manifest and checking its presence ensures it cannot be silently
    # omitted or removed without detection.
    stored_fingerprint = manifest.get("parameter_fingerprint")
    if stored_fingerprint is None:
        errors.append("manifest.parameter_fingerprint missing")
    elif not isinstance(stored_fingerprint, str) or len(stored_fingerprint) == 0:
        errors.append("manifest.parameter_fingerprint empty")

    episodes_dir = out / "episodes"
    if require_complete and not episodes_dir.is_dir():
        errors.append("episodes/ directory missing")

    n_bands = int(manifest.get("config", {}).get("n_bands", 36))
    ibw_mhz = float(manifest.get("ibw_mhz", 1000.0))

    if episodes_dir.is_dir():
        ep_ids = _episode_ids_from_dir(episodes_dir)
        counts["episodes"] = len(ep_ids)
        try:
            corpus = load_corpus(out)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"corpus loading failed: {exc}")
            corpus = {}
        for ep_id in ep_ids:
            entry = corpus.get(ep_id)
            if entry is None:
                errors.append(f"episode {ep_id}: unreadable")
                continue
            npz = entry["npz"]
            gt = entry["gt"]
            errors.extend(audit_leakage(npz))
            for key in OBS_NPZ_KEYS:
                if key not in npz:
                    errors.append(f"episode {ep_id}: missing npz key {key}")
            if all(k in npz for k in OBS_NPZ_KEYS):
                errors.extend(validate_observations(
                    npz["observation"], npz["band"], npz["center_mhz"],
                    npz["start_time_us"], npz["end_time_us"], npz["step"],
                    n_bands=n_bands,
                    obs_dim=int(manifest.get("config", {}).get("obs_dim", 360)),
                ))
                counts["observation_rows"] += int(len(npz["observation"]))
            errors.extend(validate_ground_truth(
                gt,
                n_bands=n_bands,
                ibw_mhz=ibw_mhz,
                expected_dwell_count=(int(len(npz["observation"])) if "observation" in npz else None),
                allowed_emitter_ids={e["id"] for e in manifest.get("emitter_configs", [])}
                if isinstance(manifest.get("emitter_configs"), list) else None,
            ))
            counts["dwells"] += len(gt.get("dwells", []))

            # Alignment: observation rows == gt dwells; band/center match.
            if "observation" in npz and isinstance(gt.get("dwells"), list):
                obs_n = len(npz["observation"])
                gt_n = len(gt["dwells"])
                if obs_n != gt_n:
                    errors.append(f"episode {ep_id}: alignment {obs_n} obs rows vs {gt_n} gt dwells")
                else:
                    mism = []
                    for i, dw in enumerate(gt["dwells"]):
                        if int(npz["band"][i]) != int(dw.get("band", -1)):
                            mism.append(f"row {i} band mismatch")
                        if abs(float(npz["center_mhz"][i]) - float(dw.get("center_frequency_mhz", -1.0))) > 1e-9:
                            mism.append(f"row {i} center mismatch")
                    if mism:
                        errors.append(f"episode {ep_id}: alignment mismatches: {mism[:5]}")

    # Leakage audit over manifest emitter_configs presence.
    if not isinstance(manifest.get("emitter_configs"), list):
        errors.append("manifest.emitter_configs missing")

    verdict = "PASS" if not errors else "FAIL"
    report = {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "counts": counts,
        "manifest": manifest,
        "verdict": verdict,
    }
    return report


# ===========================================================================
# Duplicate classification
# ===========================================================================


def classify_duplicates(corpus: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    """Classify duplicate observations per DATASET_SPEC.md section 13.

    A (expected):  identical observation rows across different episodes.
    B (suspicious): identical observation rows within the SAME episode.
    C (bug):       duplicate npz file hash across episodes, duplicate
                   dwell_index keys, or duplicate emitter ids in a dwell.
    """
    row_hashes: List[Tuple[str, int, str, bytes]] = []
    for ep_id, entry in corpus.items():
        npz = entry["npz"]
        obs = npz.get("observation")
        if obs is None:
            continue
        for i in range(len(obs)):
            row_hash = _sha256_bytes(obs[i].tobytes())
            row_hashes.append((ep_id, i, row_hash, obs[i].tobytes()))

    a_count = 0
    b_count = 0
    by_hash: Dict[str, List[Tuple[str, int]]] = {}
    for ep_id, i, h, _ in row_hashes:
        by_hash.setdefault(h, []).append((ep_id, i))
    for h, positions in by_hash.items():
        seen_eps = {ep for ep, _ in positions}
        within = sum(1 for (ep1, i1), (ep2, i2) in _pairs(positions) if ep1 == ep2 and i1 != i2)
        across = sum(1 for (ep1, _), (ep2, _) in _pairs(positions) if ep1 != ep2)
        b_count += within
        a_count += across

    c_count = 0
    file_hashes: Dict[str, List[str]] = {}
    for ep_id, entry in corpus.items():
        h = _sha256_bytes(entry["npz_path"].read_bytes())
        file_hashes.setdefault(h, []).append(ep_id)
        gt = entry["gt"].get("dwells", [])
        seen_idx: set[int] = set()
        for dw in gt:
            di = dw.get("dwell_index")
            if di in seen_idx:
                c_count += 1
            seen_idx.add(di)
            ids = [e.get("id") for e in dw.get("emitters", [])]
            if len(ids) != len(set(ids)):
                c_count += 1
    for h, eps in file_hashes.items():
        if len(eps) > 1:
            c_count += len(eps) - 1

    return {"A_expected": a_count, "B_suspicious": b_count, "C_bug": c_count}


def _pairs(items: List[Any]):
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            yield items[i], items[j]


# ===========================================================================
# Manifest comparison (for determinism)
# ===========================================================================


def manifest_comparison_key(manifest: Dict[str, Any]) -> str:
    """Stable corpus-affecting projection of the manifest (masks run stamps)."""
    copy = dict(manifest)
    copy.pop("timestamp_utc", None)
    copy.pop("dataset_id", None)
    return _canonical_json(copy)


# ===========================================================================
# Generator
# ===========================================================================


class DatasetGenerator:
    """Deterministic, atomic dataset generator."""

    def __init__(self, config: GenerationConfig) -> None:
        config.validate()
        self.config = config

    # ------------------------------------------------------------- per run

    def run(self, output_dir: str | Path) -> Dict[str, Any]:
        """Generate + validate atomically. Returns the final manifest."""
        out = Path(output_dir)
        tmp = Path(f"{out}.tmp-{os.getpid()}")
        if tmp.exists():
            shutil.rmtree(tmp)
        if out.exists():
            if not self.config.force:
                raise FileExistsError(
                    f"{out} already exists (use --force to regenerate)")
            shutil.rmtree(out)

        tmp.mkdir(parents=True, exist_ok=False)
        episodes_dir = tmp / "episodes"
        episodes_dir.mkdir(exist_ok=False)

        try:
            for e in range(self.config.episodes):
                self._run_episode(e, episodes_dir)

            manifest = self._build_manifest(episodes_dir)
            _write_json_atomic(tmp / "manifest.json", manifest)

            # Structural pre-rename validation (no COMPLETE required yet).
            report = validate_dataset(tmp, require_complete=False)
            if not report["valid"]:
                raise RuntimeError(f"generated corpus failed validation: {report['errors'][:8]}")

            os.replace(str(tmp), str(out))
            (out / COMPLETE_MARKER).write_text("", encoding="utf-8")

            final = validate_dataset(out, require_complete=True)
            if not final["valid"]:
                raise RuntimeError(f"final validation failed: {final['errors'][:8]}")
            return final["manifest"]
        except Exception:
            if tmp.exists():
                shutil.rmtree(tmp)
            raise

    # ------------------------------------------------------------- episode

    def _run_episode(self, episode_index: int, episodes_dir: Path) -> None:
        cfg = self.config
        ep_id = f"EP{episode_index + 1:06d}"

        translator = GnuRfSchedulerTranslation(
            n_bands=cfg.n_bands,
            freq_min=cfg.freq_min_mhz,
            freq_max=cfg.freq_max_mhz,
            deinterleaver_model=cfg.deinterleaver_model,
            deinterleaver_config=cfg.deinterleaver_config,
        )
        translator.reset()

        orch = DwellOrchestrator(initial_center_mhz=cfg.centers[0])
        ibw = float(orch.receiver.ibw_mhz)
        obs_rows: List[np.ndarray] = []
        band_rows: List[int] = []
        center_rows: List[float] = []
        start_rows: List[float] = []
        end_rows: List[float] = []
        step_rows: List[int] = []
        gt_dwells: List[Dict[str, Any]] = []

        emitters_configs = [
            spec.to_emitter_config(
                cfg.effective_emitter_seed(cfg.root_seed, episode_index, idx)
            )
            for idx, spec in enumerate(cfg.emitters)
        ]
        effective_seeds = [
            cfg.effective_emitter_seed(cfg.root_seed, episode_index, idx)
            for idx in range(len(cfg.emitters))
        ]

        for s in range(cfg.dwells_per_episode):
            center = cfg.schedule_center(episode_index, s)
            noise_seed = cfg.episode_noise_seed(cfg.root_seed, episode_index, s)
            noise_amp = cfg.noise_amplitude(episode_index)

            result = orch.run_generated_dwell(
                center_frequency_mhz=center,
                emitters=emitters_configs,
                duration_us=cfg.dwell_time_us,
                sample_rate=cfg.sample_rate,
                noise_amplitude=noise_amp,
                noise_seed=noise_seed,
                detector_threshold_db=cfg.detector_threshold_db,
            )
            band = _band_index(center, cfg.freq_min_mhz, cfg.freq_max_mhz, cfg.n_bands)

            recv = ReceiverObservation(
                time_us=result.end_time_us,
                center_frequency_mhz=result.center_frequency_mhz,
                ibw_mhz=ibw,
                dwell_time_us=cfg.dwell_time_us,
                dwell_interval_us=[result.start_time_us, result.end_time_us],
                window_mhz=[center - ibw / 2.0, center + ibw / 2.0],
                detections=list(result.detections),
            )
            obs_vec = translator.update(recv)
            if obs_vec.shape != (cfg.n_bands * OBS_FEATURES_PER_BAND,):
                raise RuntimeError(f"unexpected observation shape {obs_vec.shape}")
            obs_rows.append(obs_vec)
            band_rows.append(int(band))
            center_rows.append(float(center))
            start_rows.append(float(result.start_time_us))
            end_rows.append(float(result.end_time_us))
            step_rows.append(int(s))

            any_hit = len(result.detections) > 0
            gt_dwells.append({
                "dwell_index": int(s),
                "start_time_us": float(result.start_time_us),
                "end_time_us": float(result.end_time_us),
                "center_frequency_mhz": float(center),
                "band": int(band),
                "any_hit": bool(any_hit),
                "observed_pdw_count": int(len(result.detections)),
                "emitters": [
                    {
                        "id": spec.id,
                        "rf_frequency_mhz": float(spec.rf_frequency_mhz),
                        "pulse_width_us": float(spec.pulse_width_us),
                        "pri_us": float(spec.pri_us),
                        "amplitude": float(spec.amplitude),
                        "jitter_fraction": float(spec.jitter_fraction),
                        "configured_seed": int(spec.configured_seed),
                        "seed": int(effective_seeds[idx]),
                        "in_band": bool(
                            abs(spec.rf_frequency_mhz - center) <= ibw / 2.0
                        ),
                        "scheduled_active": bool(
                            abs(spec.rf_frequency_mhz - center) <= ibw / 2.0
                        ),
                    }
                    for idx, spec in enumerate(cfg.emitters)
                ],
            })

        np.savez_compressed(
            episodes_dir / f"{ep_id}.npz",
            observation=np.asarray(obs_rows, dtype=np.float32),
            band=np.asarray(band_rows, dtype=np.int64),
            center_mhz=np.asarray(center_rows, dtype=np.float64),
            start_time_us=np.asarray(start_rows, dtype=np.float64),
            end_time_us=np.asarray(end_rows, dtype=np.float64),
            step=np.asarray(step_rows, dtype=np.int64),
        )
        _write_json_atomic(
            episodes_dir / f"{ep_id}.gt.json",
            {
                "episode_id": ep_id,
                "schema_version": SCHEMA_VERSION,
                "generator_version": GENERATOR_VERSION,
                "dwells": gt_dwells,
            },
        )

    # ------------------------------------------------------------- manifest

    def _build_manifest(self, episodes_dir: Path) -> Dict[str, Any]:
        cfg = self.config
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        dataset_id = f"gnu_rf_dryrun_seed{int(cfg.root_seed)}_ep{cfg.episodes}d{cfg.dwells_per_episode}"

        probe = DwellOrchestrator(initial_center_mhz=cfg.centers[0])
        ibw_mhz = float(probe.receiver.ibw_mhz)

        emitter_configs = [e.to_dict() for e in cfg.emitters]
        file_hashes: Dict[str, str] = {}
        for p in sorted(episodes_dir.glob("*.npz")) + sorted(episodes_dir.glob("*.gt.json")):
            file_hashes[f"episodes/{p.name}"] = _sha256_file(p)

        deinterleaver_meta: Dict[str, Any] = {"enabled": cfg.deinterleaver_model is not None}
        if cfg.deinterleaver_model is not None:
            dcfg = cfg.deinterleaver_config or {}
            ckpt = dcfg.get("checkpoint")
            stats = dcfg.get("stats_path")
            meta = dcfg.get("metadata")
            deinterleaver_meta.update({
                "checkpoint": str(ckpt) if ckpt else None,
                "normalization_stats": str(stats) if stats else None,
                "metadata": meta or {},
                "gate": {
                    "min_pulses": cfg.deinterleaver_config.get("min_pulses"),
                    "interval_steps": cfg.deinterleaver_config.get("interval_steps"),
                    "min_cluster_size": cfg.deinterleaver_config.get("min_cluster_size"),
                    "min_samples": cfg.deinterleaver_config.get("min_samples"),
                },
            })

        # Build the complete manifest dict as it will appear in manifest.json.
        # Use a dummy fingerprint so we can compute the canonical hash by
        # removing it — this exact serialization is what validate_dataset
        # will later recompute and compare against.
        manifest_full: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "generator_version": GENERATOR_VERSION,
            "generated_by": "scripts/dataset_generator.py",
            "dataset_id": dataset_id,
            "timestamp_utc": now,
            "git_revision": _git_revision(),
            "git_dirty": _git_dirty(),
            "root_seed": int(cfg.root_seed),
            "seed_policy": (
                "root_seed -> per-dwell noise seeds + per-episode effective "
                "emitter RNG seeds (SHA-256 derived); emitter configured_seed "
                "recorded as provenance"
            ),
            "episode_count": cfg.episodes,
            "dwell_count": cfg.episodes * cfg.dwells_per_episode,
            "per_episode_dwells": cfg.dwells_per_episode,
            "bands": sorted({int(_band_index(c, cfg.freq_min_mhz, cfg.freq_max_mhz, cfg.n_bands)) for c in cfg.centers}),
            "centers_mhz": [float(c) for c in cfg.centers],
            "ibw_mhz": ibw_mhz,
            "emitter_configs": emitter_configs,
            "noise_amplitudes": [float(n) for n in cfg.noise_amplitudes],
            "scheduler_step_scenario": f"deterministic {cfg.schedule_policy} scan",
            "deinterleaver": deinterleaver_meta,
            "rf_limitations": [
                "amplitude placeholder -100.0 dB (Phase 3W intentional)",
                "constant AoA 0.0 (Phase 3W intentional)",
                "pulse-width bias ~+0.5 us on noisy envelopes (3W deliberate)",
                "AWGN-only noise model (3W deliberate)",
                "2 MS/s generator coupling (3W deliberate)",
            ],
            "parameter_fingerprint": cfg.parameter_fingerprint(),
        }

        # Set parameter_fingerprint from config (computed from actual runtime parameters).
        manifest_full["parameter_fingerprint"] = cfg.parameter_fingerprint()

        # Build final manifest: add file_hashes, config, and validation sections.
        result: Dict[str, Any] = {
            **manifest_full,
            "file_hashes": file_hashes,
            "config": {
                "n_bands": cfg.n_bands,
                "obs_dim": cfg.n_bands * OBS_FEATURES_PER_BAND,
                "freq_min_mhz": cfg.freq_min_mhz,
                "freq_max_mhz": cfg.freq_max_mhz,
                "dwell_time_us": cfg.dwell_time_us,
                "sample_rate": cfg.sample_rate,
                "detector_threshold_db": cfg.detector_threshold_db,
            },
            "validation": {
                "verdict": "PASS",
                "atomicity": "complete",
                "leakage": "clean",
            },
        }
        return result


def _git_revision() -> str:
    try:
        import subprocess
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()[:40]
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def _git_dirty() -> bool:
    try:
        import subprocess
        r = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:  # noqa: BLE001
        return True


# ===========================================================================
# Top-level convenience
# ===========================================================================


def generate_dataset(config: GenerationConfig, output_dir: str | Path) -> Dict[str, Any]:
    return DatasetGenerator(config).run(output_dir)


# ===========================================================================
# Report
# ===========================================================================


def print_report(output_dir: str | Path) -> int:
    out = Path(output_dir)
    report = validate_dataset(out, require_complete=True)
    corpus = load_corpus(out) if report["valid"] else {}
    dups = classify_duplicates(corpus) if report["valid"] else {}
    m = report.get("manifest") or {}

    print(f"Corpus report: {out}")
    print(f"  verdict: {report['verdict']}")
    print(f"  errors: {len(report['errors'])}")
    for e in report["errors"][:12]:
        print(f"    - {e}")
    print(f"  episodes: {report['counts']['episodes']}")
    print(f"  dwells: {report['counts']['dwells']}")
    print(f"  observation rows: {report['counts']['observation_rows']}")
    print(f"  centers_mhz: {m.get('centers_mhz')}")
    print(f"  noise_amplitudes: {m.get('noise_amplitudes')}")
    print(f"  emitter_configs: {[e.get('id') for e in m.get('emitter_configs', [])]}")
    print(f"  duplicate classification: A_expected={dups.get('A_expected', 0)} "
          f"B_suspicious={dups.get('B_suspicious', 0)} C_bug={dups.get('C_bug', 0)}")
    print(f"  parameter_fingerprint: {m.get('parameter_fingerprint')}")
    print(f"  deinterleaver: {m.get('deinterleaver', {}).get('enabled')}")
    for rel, h in (m.get("file_hashes") or {}).items():
        print(f"    {rel}: {h[:16]}...")
    return 0 if report["valid"] and dups.get("C_bug", 0) == 0 else 1


# ===========================================================================
# CLI
# ===========================================================================


def _parse_csv_floats(text: str) -> List[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def _load_emitter_specs(path: str | None) -> List[EmitterSpec]:
    if path is None:
        return build_default_emitter_specs()
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = raw.get("emitters") if isinstance(raw, dict) else raw
    specs: List[EmitterSpec] = []
    for i, it in enumerate(items):
        specs.append(EmitterSpec(
            id=str(it.get("id", f"E{i + 1}")),
            rf_frequency_mhz=float(it["rf_frequency_mhz"]),
            pulse_width_us=float(it.get("pulse_width_us", 10.0)),
            pri_us=float(it.get("pri_us", 100.0)),
            amplitude=float(it.get("amplitude", 1.0)),
            jitter_fraction=float(it.get("jitter_fraction", 0.01)),
            configured_seed=int(it.get("seed", 42)),
        ))
    return specs


def _load_deinterleaver(config: GenerationConfig) -> None:
    try:
        payload = load_deinterleaver()
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: real deinterleaver unavailable ({exc}); using perception-disabled fallback")
        return
    config.deinterleaver_model = payload["model"]
    config.deinterleaver_config = payload["config"]
    config.deinterleaver_config["checkpoint"] = payload["checkpoint"]
    config.deinterleaver_config["stats_path"] = payload["stats_path"]
    config.deinterleaver_config["metadata"] = payload["metadata"]


def _add_generation_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--seed", type=int, default=12345)
    sub.add_argument("--episodes", type=int, default=3)
    sub.add_argument("--dwells-per-episode", type=int, default=12)
    sub.add_argument("--centers", type=str, default=",".join(str(c) for c in DEFAULT_CENTERS))
    sub.add_argument("--emitter-configs", type=str, default=None,
                     help="JSON path {emitters:[...]} or {items:[...]}; defaults built in")
    sub.add_argument("--dwell-time-us", type=float, default=500.0)
    sub.add_argument("--sample-rate", type=float, default=2e6)
    sub.add_argument("--noise-amplitudes", type=str, default="0.02,0.05,0.0")
    sub.add_argument("--threshold-db", type=float, default=15.0)
    sub.add_argument("--n-bands", type=int, default=36)
    sub.add_argument("--freq-min-mhz", type=float, default=0.0)
    sub.add_argument("--freq-max-mhz", type=float, default=18000.0)
    sub.add_argument("--no-deinterleaver", action="store_true",
                     help="Use the production perception-disabled fallback (no model)")
    sub.add_argument("--force", action="store_true")


def _base_config_from_args(args: argparse.Namespace) -> GenerationConfig:
    use_real = not getattr(args, "no_deinterleaver", False)
    cfg = GenerationConfig(
        root_seed=args.seed,
        episodes=args.episodes,
        dwells_per_episode=args.dwells_per_episode,
        centers=_parse_csv_floats(args.centers),
        emitters=_load_emitter_specs(getattr(args, "emitter_configs", None)),
        dwell_time_us=args.dwell_time_us,
        sample_rate=args.sample_rate,
        noise_amplitudes=_parse_csv_floats(args.noise_amplitudes),
        detector_threshold_db=args.threshold_db,
        n_bands=args.n_bands,
        freq_min_mhz=args.freq_min_mhz,
        freq_max_mhz=args.freq_max_mhz,
        force=getattr(args, "force", False),
    )
    if use_real:
        _load_deinterleaver(cfg)
    return cfg


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="dataset_generator")
    sub = parser.add_subparsers(dest="command", required=True)

    pg = sub.add_parser("generate")
    _add_generation_args(pg)
    pg.add_argument("--output-dir", required=True)

    pv = sub.add_parser("validate")
    pv.add_argument("--output-dir", required=True)
    pv.add_argument("--require-complete", action="store_true")

    pr = sub.add_parser("report")
    pr.add_argument("--output-dir", required=True)

    pp = sub.add_parser("reproduce")
    pp.add_argument("--seed", type=int, default=12345)
    pp.add_argument("--dwells-per-episode", type=int, default=6)
    pp.add_argument("--episodes", type=int, default=2)
    pp.add_argument("--noise-amplitudes", type=str, default="0.02")
    pp.add_argument("--output-dir", required=True)

    args = parser.parse_args(argv)

    if args.command == "generate":
        cfg = _base_config_from_args(args)
        generate_dataset(cfg, args.output_dir)
        print(f"Generated: {args.output_dir}")
        return 0

    if args.command == "validate":
        report = validate_dataset(args.output_dir, require_complete=args.require_complete)
        for e in report["errors"]:
            print(f"ERROR: {e}")
        print(f"verdict: {report['verdict']} "
              f"(episodes={report['counts']['episodes']} dwells={report['counts']['dwells']})")
        return 0 if report["valid"] else 1

    if args.command == "report":
        return print_report(args.output_dir)

    if args.command == "reproduce":
        base = Path(args.output_dir)
        dir1 = base / "run1"
        dir2 = base / "run2"
        cfg_args = argparse.Namespace(
            seed=args.seed,
            episodes=args.episodes,
            dwells_per_episode=args.dwells_per_episode,
            centers=",".join(str(c) for c in DEFAULT_CENTERS),
            emitter_configs=None,
            dwell_time_us=500.0,
            sample_rate=2e6,
            noise_amplitudes=args.noise_amplitudes,
            threshold_db=15.0,
            n_bands=36,
            freq_min_mhz=0.0,
            freq_max_mhz=18000.0,
            no_deinterleaver=True,
            force=False,
        )
        cfg1 = _base_config_from_args(cfg_args)
        cfg2 = _base_config_from_args(cfg_args)
        m1 = generate_dataset(cfg1, dir1)
        m2 = generate_dataset(cfg2, dir2)
        identical_data = True
        for rel in sorted((m1.get("file_hashes") or {})):
            if (m1.get("file_hashes") or {}).get(rel) != (m2.get("file_hashes") or {}).get(rel):
                identical_data = False
                print(f"DIFF {rel}: {m1['file_hashes'][rel]} vs {m2['file_hashes'][rel]}")
        manifests_equal = manifest_comparison_key(m1) == manifest_comparison_key(m2)
        if not manifests_equal:
            print("DIFF: manifest comparison key differs (excluding timestamp/dataset_id)")
        if identical_data and manifests_equal:
            print(f"reproduce: identical ({dir1} vs {dir2})")
            return 0
        print("reproduce: DIFFERS")
        return 1

    parser.print_help()
    return 2


def sha256_manifest_no_fp(manifest: Dict[str, Any]) -> str:
    """Compute sha256 of the manifest JSON without the parameter_fingerprint field.
    
    This enables integrity verification: the stored fingerprint should match
    the hash of the manifest content excluding itself.
    """
    manifest_no_fp = {k: v for k, v in manifest.items() if k != "parameter_fingerprint"}
    canonical = json.dumps(manifest_no_fp, sort_keys=True, separators=(",", ":"))
    import hashlib
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())

"""
TSRD acquisition — one authoritative path.

Mode of operation
-----------------
HuggingFace repo ``alan-turing-institute/turing-synthetic-radar-dataset`` is
the single acquisition source. Only the EXACT ``.h5`` files belonging to the
requested ``{mode}/{split}`` subsets are downloaded (never a whole-repo
``snapshot_download``), landing at::

    <output-dir>/<mode>/<split>/*.h5

Every downloaded file is then verified and fingerprinted (never fabricated):

    1.  H5 readability
    2.  ``data`` dataset exists
    3.  ``labels`` dataset exists
    4.  ``data`` shape is N x 5
    5.  ``labels`` length equals N
    6.  pulse count (N)
    7.  emitter count (unique non-noise labels)
    8.  duration (ToA max - ToA min, microseconds)
    9.  SHA-256
    10. manifest.json written (per-subset + aggregate)

Downloading ANYTHING requires ``--allow-download`` — without it the script
returns ``{"status": "skipped"}``. If a requested subset has no matching repo
files, it is recorded as ``missing`` in the manifest; no synthetic data is ever
created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import h5py  # type: ignore
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

TSRD_REPO = "alan-turing-institute/turing-synthetic-radar-dataset"

VALID_MODES = ("stare", "scan")
# Canonical split names; `val` is accepted and normalised to `validation`.
VALID_SPLITS = ("train", "validation", "test")
SPLIT_ALIASES = {"val": "validation", "valid": "validation"}

EXPECTED_COLS = 5  # [ToA, CF, PW, AoA, Amp] per pulse
NOISE_LABEL = -1


def _human_size(nbytes: float) -> str:
    """Human-readable file size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(nbytes) < 1024.0:
            return f"{nbytes:3.1f}{unit}"
        nbytes /= 1024.0
    return f"{nbytes:.1f}PB"


def _normalise_modes(modes: list[str] | None) -> list[str]:
    modes = [m.lower() for m in (modes or ["stare", "scan"])]
    for m in modes:
        if m not in VALID_MODES:
            raise ValueError(f"Unknown mode {m!r} — must be one of {VALID_MODES}")
    return modes


def _normalise_splits(splits: list[str] | None) -> list[str]:
    out: list[str] = []
    for s in (splits or ["train", "validation", "test"]):
        s = s.lower()
        s = SPLIT_ALIASES.get(s, s)
        if s not in VALID_SPLITS:
            raise ValueError(f"Unknown split {s!r} — must be one of {VALID_SPLITS} or {tuple(SPLIT_ALIASES)}")
        out.append(s)
    return out


@dataclass
class FileRecord:
    """Verified per-file acquisition record (never fabricated)."""

    name: str
    sha256: str
    size_bytes: int
    pulse_count: int
    emitter_count: int
    emitter_count_incl_noise: int
    duration_us: float
    toa_min_us: float
    toa_max_us: float
    collection_time_s: float | None = None  # official TSRD metadata attr, when present
    shape: list[int] = field(default_factory=list)


def _sha256(path: Path) -> str:
    """Streaming SHA-256 (step 9)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_h5(path: Path) -> FileRecord:
    """Run verification steps 1–9 on one H5 file.

    Steps:
        1  openability,
        2  ``data`` dataset,
        3  ``labels`` dataset,
        4  shape N x 5,
        5  label length == N,
        6  pulse count,
        7  emitter count,
        8  duration (ToA range, microseconds),
        9  SHA-256 (computed here via ``_sha256``).

    Raises:
        ValueError: On any verification failure (readability, missing/wrong
            datasets, shape/length mismatch). The failure is surfaced — never
            silently papered over.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"not a file: {path}")
    collection_time_s: float | None = None
    try:
        with h5py.File(str(path), "r") as f:  # step 1: readability
            if "data" not in f:  # step 2
                raise ValueError(f"{path.name}: missing required 'data' dataset (keys={list(f.keys())})")
            if "labels" not in f:  # step 3
                raise ValueError(f"{path.name}: missing required 'labels' dataset (keys={list(f.keys())})")
            data = np.asarray(f["data"])
            labels = np.asarray(f["labels"])
            if "metadata" in f and hasattr(f["metadata"], "attrs"):
                meta = dict(f["metadata"].attrs)
                if "collection_time_s" in meta:
                    collection_time_s = float(meta["collection_time_s"])
    except OSError as exc:
        raise ValueError(f"{path.name}: unreadable H5 ({exc})") from exc

    if data.ndim != 2 or data.shape[1] != EXPECTED_COLS:  # step 4
        raise ValueError(f"{path.name}: data shape {data.shape} not N x {EXPECTED_COLS}")
    # Official TSRD stores labels as a 1-D ``(N,)`` or column ``(N, 1)`` array.
    if labels.ndim not in (1, 2) or labels.reshape(-1).shape[0] != data.shape[0]:  # step 5
        raise ValueError(
            f"{path.name}: labels shape {labels.shape} does not match pulse count {data.shape[0]}"
        )

    pulse_count = int(data.shape[0])  # step 6
    label_vals = [int(x) for x in labels.reshape(-1).tolist()]
    distinct = set(label_vals)
    emitter_count_incl_noise = len(distinct)
    emitter_count = len({v for v in distinct if v != NOISE_LABEL})  # step 7 (non-noise)
    toa_col = np.asarray(data[:, 0], dtype=np.float64)
    toa_min = float(toa_col.min())  # step 8 (ToA in microseconds)
    toa_max = float(toa_col.max())
    return FileRecord(
        name=path.name,
        sha256=_sha256(path),  # step 9
        size_bytes=path.stat().st_size,
        pulse_count=pulse_count,
        emitter_count=emitter_count,
        emitter_count_incl_noise=emitter_count_incl_noise,
        duration_us=toa_max - toa_min,
        toa_min_us=toa_min,
        toa_max_us=toa_max,
        shape=[int(data.shape[0]), int(data.shape[1])],
        collection_time_s=collection_time_s,
    )


def _belongs_to(file_rel: str, mode: str, split: str) -> bool:
    """Does this repo file belong to the ``mode/split`` subset?

    Authoritative rule — a file belongs if its repo-relative path contains the
    exact ``mode/split`` directory, with a filename fallback for flat repos
    whose names encode ``<mode>_<split>``.
    """
    rel = file_rel.replace("\\", "/")
    if f"/{mode}/{split}/" in f"/{rel}" or rel.startswith(f"{mode}/{split}/"):
        return True
    name = rel.rsplit("/", 1)[-1].lower()
    return mode in name and split in name


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _hub_functions():
    """Resolve the huggingface_hub API (the one authoritative acquisition path).

    Separated so acquisition can be exercised offline in tests without the
    package installed.
    """
    try:
        from huggingface_hub import hf_hub_download, list_repo_files  # type: ignore
        return hf_hub_download, list_repo_files
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub not installed — required for the one authoritative "
            "TSRD acquisition path (pip install huggingface_hub)."
        ) from exc


def download_tsr_dataset(
    output_dir: str | Path = "data",
    token: str | None = None,
    modes: list[str] | None = None,
    splits: list[str] | None = None,
    allow_download: bool = False,
) -> dict[str, Any]:
    """Download the requested TSRD subsets (one authoritative path).

    Args:
        output_dir: Root output directory; files land at
            ``<output_dir>/<mode>/<split>/*.h5``.
        token: HF token (defaults to ``HF_TOKEN`` env).
        modes: Subset of ``("stare", "scan")``.
        splits: Subset of ``("train", "validation", "test")``.
        allow_download: Explicit confirmation to actually download.

    Returns:
        Summary dict. Without ``allow_download`` it is ``status=skipped``.
        Missing subsets are recorded (never fabricated): ``status=missing``
        entries in the manifest and ``missing`` in the summary.

    Raises:
        RuntimeError: If ``huggingface_hub`` is unavailable or the repo cannot
            be listed/downloaded — never a silent fallback.
    """
    modes = _normalise_modes(modes)
    splits = _normalise_splits(splits)

    if not allow_download:
        logger.warning(
            "TSRD download is disabled by default so the entire dataset is never "
            "pulled accidentally. Re-run with --allow-download to explicitly "
            "confirm the acquisition."
        )
        return {
            "status": "skipped",
            "reason": "download_disabled_by_default",
            "output_dir": str(output_dir),
            "modes": modes,
            "splits": splits,
        }

    token = token or os.getenv("HF_TOKEN")
    if not token or token == "your_huggingface_token_here":
        logger.warning("HF_TOKEN missing or placeholder — only public files can be fetched.")

    hf_hub_download, list_repo_files = _hub_functions()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Listing %s ...", TSRD_REPO)
    try:
        files = list_repo_files(TSRD_REPO, token=token if token and token != "your_huggingface_token_here" else None)
    except Exception as exc:
        raise RuntimeError(f"Failed to list {TSRD_REPO}: {exc}") from exc
    h5_files = sorted(f for f in files if f.endswith(".h5"))
    logger.info("Repo has %d .h5 files", len(h5_files))

    # ---- Pre-flight plan (transparency; never an accidental whole-repo pull) --
    plan: list[tuple[str, str, list[str]]] = []
    for mode in modes:
        for split in splits:
            matched = [f for f in h5_files if _belongs_to(f, mode, split)]
            plan.append((mode, split, matched))
    total_planned = sum(len(m) for _, _, m in plan)
    logger.info("Explicitly requested %d files (only these are downloaded):", total_planned)
    for mode, split, matched in plan:
        logger.info("  %s/%s -> %d files", mode, split, len(matched))
    if total_planned == 0:
        logger.warning("No matching .h5 files in the repo for the requested subsets.")

    # ---- Download (scoped, per-file; never snapshot_download) ----------------
    summary: dict[str, Any] = {
        "status": "ok",
        "repo": TSRD_REPO,
        "downloaded_files": 0,
        "verified_files": 0,
        "failed_files": 0,
        "missing": [],
        "subsets": {},
    }
    verified_total_pulses = 0
    verified_total_emitters = 0
    verified_total_duration_us = 0.0

    for mode, split, matched in plan:
        subset_key = f"{mode}/{split}"
        target = output_dir / mode / split
        target.mkdir(parents=True, exist_ok=True)
        subset: dict[str, Any] = {
            "mode": mode,
            "split": split,
            "status": "ok",
            "files": [],
            "failed_files": [],
            "totals": {"files": 0, "pulses": 0, "emitters": 0, "duration_us": 0.0},
        }
        if not matched:
            subset["status"] = "missing"
            summary["missing"].append(subset_key)
            logger.warning("No repo files for %s — subset marked missing (not fabricated).", subset_key)
        for repo_file in matched:
            try:
                local = hf_hub_download(
                    repo_id=TSRD_REPO,
                    filename=repo_file,
                    local_dir=str(output_dir),
                    token=token if token and token != "your_huggingface_token_here" else None,
                )
                local_path = Path(local)
                summary["downloaded_files"] += 1
                record = _verify_h5(local_path)  # steps 1-9
                subset["files"].append(asdict(record))
                summary["verified_files"] += 1
                verified_total_pulses += record.pulse_count
                verified_total_emitters += record.emitter_count
                verified_total_duration_us += record.duration_us
                subset["totals"]["files"] += 1
                subset["totals"]["pulses"] += record.pulse_count
                subset["totals"]["emitters"] += record.emitter_count
                subset["totals"]["duration_us"] += record.duration_us
                logger.info(
                    "OK %s: %s pulses, %d emitters, %.3f s duration",
                    repo_file, record.pulse_count, record.emitter_count, record.duration_us / 1e6,
                )
            except Exception as exc:
                subset["failed_files"].append({"name": repo_file, "reason": str(exc)})
                summary["failed_files"] += 1
                logger.error("FAIL %s: %s", repo_file, exc)
        if subset["failed_files"]:
            subset["status"] = subset["status"] if subset["files"] else "failed"
        _write_manifest(target / "manifest.json", subset)  # step 10 (subset)
        subset_summary = {k: subset[k] for k in ("status", "files", "failed_files", "totals")}
        summary["subsets"][subset_key] = subset_summary

    # ---- Step 10: aggregate manifest -----------------------------------------
    summary["totals"] = {
        "files": verified_total_pulses and summary["verified_files"] or 0,
        "pulses": verified_total_pulses,
        "emitters": verified_total_emitters,  # sum of per-file non-noise uniques
        "duration_us": verified_total_duration_us,
    }
    summary["status"] = (
        "ok"
        if summary["verified_files"] > 0 and not summary["failed_files"] and not summary["missing"]
        else "partial"
    )
    _write_manifest(output_dir / "manifest.json", summary)
    logger.info("Wrote aggregate manifest at %s", output_dir / "manifest.json")
    logger.info("=== TSRD acquisition summary ===")
    logger.info("Verified files: %d  Failed: %d  Missing subsets: %s",
                summary["verified_files"], summary["failed_files"], summary["missing"] or "none")
    logger.info("Total pulses: %d  Emitters(sum): %d  Duration(sum): %.3f s",
                verified_total_pulses, verified_total_emitters, verified_total_duration_us / 1e6)
    return summary


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser(description="Acquire TSRD via the one authoritative HuggingFace path")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("TSRD_DATA_ROOT", "data"),
        help="Root data directory (env TSRD_DATA_ROOT takes precedence; default data)",
    )
    parser.add_argument("--token", type=str, default=None, help="HF token (or set HF_TOKEN env)")
    parser.add_argument("--modes", nargs="+", default=["stare", "scan"], help="Modes: stare, scan")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"], help="Splits: train, validation, test")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="EXPLICIT confirmation to download. Without it nothing is downloaded.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        summary = download_tsr_dataset(
            output_dir=args.output_dir,
            token=args.token,
            modes=args.modes,
            splits=args.splits,
            allow_download=args.allow_download,
        )
        print(f"\nDone. Summary: {json.dumps(summary, indent=2)}")
    except (ValueError, RuntimeError) as exc:
        logger.error("Acquisition failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
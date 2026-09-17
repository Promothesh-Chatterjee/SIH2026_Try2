"""
TSRD acquisition — one authoritative path.

Mode of operation
-----------------
HuggingFace repo ``alan-turing-institute/turing-synthetic-radar-dataset`` is
the single acquisition source. Only the EXACT ``.h5`` files belonging to the
requested ``{mode}/{split}`` subsets are downloaded (never a whole-repo
``snapshot_download``), landing at::

    <output-dir>/<mode>/<split>/*.h5

Both the conventional split directory names (``scan/train``,
``scan/validation`` …) and the official Kaggle names (``train_scan``,
``val_scan``, ``test_scan``, ``train_stare`` …) are recognised when matching
repo files, so a Kaggle-layout repo is downloaded into the canonical
``<mode>/<split>/`` tree.

Every downloaded file is then verified and fingerprinted (never fabricated):

    1.  H5 readability
    2.  ``data`` dataset exists
    3.  ``labels`` dataset exists
    4.  ``data`` shape is N x 5
    5.  ``labels`` length equals N
    6.  all values finite
    7.  ToA non-decreasing (per-file ordering)
    8.  pulse count (N)
    9.  emitter count (unique non-noise labels)
    10. duration (ToA max - ToA min, microseconds)
    11. SHA-256
    12. manifest.json written (per-subset + aggregate)

Downloading ANYTHING requires ``--allow-download`` — without it the script
returns ``{"status": "skipped"}``. ``--dry-run`` lists and plans the exact file
set without writing a single byte. If a requested subset has no matching repo
files, it is recorded as ``missing`` in the manifest; no synthetic data is ever
created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import tempfile
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
# Official Kaggle dir tokens per canonical split (e.g. `train_scan`, `val_scan`).
# A split matches both its canonical token and any alias token, prefixed to the
# mode (`<token>_<mode>`) or postfixed (`<mode>_<token>`).
_SPLIT_TOKENS: dict[str, tuple[str, ...]] = {
    "train": ("train",),
    "validation": ("validation", "val", "valid"),
    "test": ("test",),
}

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
    """Run verification steps 1–11 on one H5 file.

    Steps:
        1  openability,
        2  ``data`` dataset,
        3  ``labels`` dataset,
        4  shape N x 5,
        5  label length == N,
        6  all values finite,
        7  ToA non-decreasing,
        8  pulse count,
        9  emitter count,
        10 duration (ToA range, microseconds),
        11 SHA-256 (computed here via ``_sha256``).

    Zero-pulse trains (``data (0,5)`` ``labels (0,1)``) are structurally VALID
    official TSRD scenes — they are recorded, not rejected.

    Raises:
        ValueError: On any verification failure (readability, missing/wrong
            datasets, shape/length mismatch, non-finite values, unordered ToA).
            The failure is surfaced — never silently papered over.
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
    if not np.all(np.isfinite(data)):  # step 6
        raise ValueError(f"{path.name}: data contains non-finite (NaN/Inf) values")

    pulse_count = int(data.shape[0])  # step 8
    if pulse_count > 0:
        toa_col = np.asarray(data[:, 0], dtype=np.float64)
        if np.any(np.diff(toa_col) < 0):  # step 7: ToA non-decreasing
            raise ValueError(f"{path.name}: ToA column is not non-decreasing")
        toa_min = float(toa_col.min())  # step 10 (ToA in microseconds)
        toa_max = float(toa_col.max())
        duration_us = toa_max - toa_min
    else:  # zero-pulse official scene: durations default to 0.0
        toa_min = toa_max = 0.0
        duration_us = 0.0

    label_vals = [int(x) for x in labels.reshape(-1).tolist()]
    distinct = set(label_vals)
    emitter_count_incl_noise = len(distinct)
    emitter_count = len({v for v in distinct if v != NOISE_LABEL})  # step 9 (non-noise)
    return FileRecord(
        name=path.name,
        sha256=_sha256(path),  # step 11
        size_bytes=path.stat().st_size,
        pulse_count=pulse_count,
        emitter_count=emitter_count,
        emitter_count_incl_noise=emitter_count_incl_noise,
        duration_us=duration_us,
        toa_min_us=toa_min,
        toa_max_us=toa_max,
        shape=[int(data.shape[0]), int(data.shape[1])],
        collection_time_s=collection_time_s,
    )


def _belongs_to(file_rel: str, mode: str, split: str) -> bool:
    """Does this repo file belong to the ``mode/split`` subset?

    Authoritative rule — a file belongs if its repo-relative path contains the
    exact ``mode/<split>`` directory, OR any official/alias form of it, and a
    filename fallback for flat repos whose names encode ``<mode>_<split>``.

    Directory forms accepted for a split (tokens prefixed to the mode, matching
    the official Kaggle layout):

      train        -> ``train``, ``train_scan``, ``scan_train``
      validation   -> ``validation``, ``val``, ``valid``,
                      ``validation_scan``, ``val_scan``, ``valid_scan`` … (and
                      ``scan_valid*`` postfixed variants)
      test         -> ``test``, ``test_scan``, ``scan_test``

    Examples: ``stare/train/a.h5``, ``scan/val_scan/b.h5``,
    ``stare/train_stare/c.h5`` and ``data_stare_train_001.h5`` all belong to
    their respective ``mode/split`` subsets.
    """
    rel = file_rel.replace("\\", "/")
    tokens = _SPLIT_TOKENS[split]
    # Canonical + alias directory names, in both `<token>_<mode>` (Kaggle:
    # `train_scan`) and `<mode>_<token>` orders, plus the bare token.
    dir_names = {token for token in tokens}
    for token in tokens:
        dir_names.add(f"{token}_{mode}")
        dir_names.add(f"{mode}_{token}")

    parts = rel.split("/")
    for i, part in enumerate(parts[:-1]):
        if part == mode and parts[i + 1] in dir_names:
            return True
    # Flat-repo fallback: filename encodes `<mode>_<split>` / `<mode>_<alias>`.
    name = parts[-1].lower()
    _token = (
        r"(?:^|[^a-z0-9])"
        r"{token}"
        r"(?:$|[^a-z0-9])"
    )
    has_mode = re.search(_token.format(token=re.escape(mode)), name)
    has_split = any(re.search(_token.format(token=re.escape(t)), name) for t in tokens)
    return bool(has_mode and has_split)


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
    dry_run: bool = False,
) -> dict[str, Any]:
    """Download the requested TSRD subsets (one authoritative path).

    Args:
        output_dir: Root output directory; files land at
            ``<output_dir>/<mode>/<split>/*.h5`` (canonical tree, even when the
            upstream repo uses official Kaggle names such as ``val_scan``).
        token: HF token (defaults to ``HF_TOKEN`` env).
        modes: Subset of ``("stare", "scan")``.
        splits: Subset of ``("train", "validation", "test")``.
        allow_download: Explicit confirmation to actually download.
        dry_run: List and plan the exact file set, download nothing, write no
            manifests and create no directories.

    Returns:
        Summary dict. Without ``allow_download`` it is ``status=skipped``.
        With ``dry_run`` it is ``status=dry_run``. Missing subsets are recorded
        (never fabricated): ``status=missing`` entries in the manifest and
        ``missing`` in the summary.

    Raises:
        RuntimeError: If ``huggingface_hub`` is unavailable or the repo cannot
            be listed/downloaded — never a silent fallback.
    """
    modes = _normalise_modes(modes)
    splits = _normalise_splits(splits)

    if not allow_download and not dry_run:
        logger.warning(
            "TSRD download is disabled by default so the entire dataset is never "
            "pulled accidentally. Re-run with --allow-download to explicitly "
            "confirm the acquisition (or --dry-run to plan it)."
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

    if dry_run:
        return {
            "status": "dry_run",
            "repo": TSRD_REPO,
            "downloaded_files": 0,
            "verified_files": 0,
            "would_download_files": total_planned,
            "output_dir": str(output_dir),
            "modes": modes,
            "splits": splits,
            "subsets": {f"{mode}/{split}": {"planned_files": len(m)} for mode, split, m in plan},
            "warning": "dry-run: only files matching the requested bounded subsets are matched; nothing was fetched.",
        }

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
                # Stage into a scratch cache so the canonical tree never receives
                # stray intermediate paths, then move the EXACT downloaded bytes
                # into <output>/<mode>/<split>/ (no re-encode, no rewrite).
                cache = Path(tempfile.mkdtemp(prefix="tsrd_acquire_"))
                local = hf_hub_download(
                    repo_id=TSRD_REPO,
                    filename=repo_file,
                    local_dir=str(cache),
                    token=token if token and token != "your_huggingface_token_here" else None,
                )
                local_path = Path(local)
                canonical = target / local_path.name
                shutil.move(str(local_path), str(canonical))
                shutil.rmtree(cache, ignore_errors=True)
                summary["downloaded_files"] += 1
                record = _verify_h5(canonical)  # steps 1-11
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
                    "OK %s -> %s: %s pulses, %d emitters, %.3f s duration",
                    repo_file, canonical.name, record.pulse_count, record.emitter_count, record.duration_us / 1e6,
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
    # file count is the actual verified-file counter — never derived from the
    # (possibly zero) pulse total.
    summary["totals"] = {
        "files": summary["verified_files"],
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List and plan the exact file set, download nothing and write no manifests.",
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
            dry_run=args.dry_run,
        )
        print(f"\nDone. Summary: {json.dumps(summary, indent=2)}")
    except (ValueError, RuntimeError) as exc:
        logger.error("Acquisition failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
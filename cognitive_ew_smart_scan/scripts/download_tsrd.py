"""
Download the Turing Synthetic Radar Dataset (TSRD) to a local directory.

Uses huggingface_hub.snapshot_download which is resumable, shows per-file
progress, handles the gated repo (requires HF_TOKEN), and preserves the original
repo layout (e.g. scan/train_scan/*.h5, stare/train_stare/*.h5).

Usage:
    python scripts/download_tsrd.py --output-dir D:/TSRD_data
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

REPO_ID = "alan-turing-institute/turing-synthetic-radar-dataset"


def _human_size(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024.0:
            return f"{n:.2f}{unit}"
        n /= 1024.0
    return f"{n:.2f}PB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TSRD to local dir")
    parser.add_argument("--output-dir", type=str, default="D:/TSRD_data")
    parser.add_argument("--token", type=str, default=None, help="HF read token (or set HF_TOKEN in .env)")
    parser.add_argument("--max-workers", type=int, default=6, help="parallel download workers")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    token = args.token or os.getenv("HF_TOKEN")
    if not token or token == "your_huggingface_token_here":
        raise RuntimeError(
            "HF_TOKEN is not set to a real value. "
            "Accept the gating form at the dataset page and set HF_TOKEN in .env."
        )

    from huggingface_hub import snapshot_download

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s -> %s", REPO_ID, output_dir)
    t0 = time.perf_counter()
    local_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        token=token,
        local_dir=str(output_dir),
        # keep the original scan/train_scan/... layout on disk
        local_dir_use_symlinks=False,
        # resumable + parallel for speed
        max_workers=args.max_workers,
    )
    elapsed = time.perf_counter() - t0

    # Summary
    total = 0
    n_files = 0
    for p in Path(local_dir).rglob("*"):
        if p.is_file() and p.suffix == ".h5":
            n_files += 1
            total += p.stat().st_size
    logger.info("Download complete in %.1fs", elapsed)
    logger.info("Total .h5 files: %d", n_files)
    logger.info("Total size on disk: %s", _human_size(total))
    logger.info("Local root: %s", local_dir)


if __name__ == "__main__":
    main()
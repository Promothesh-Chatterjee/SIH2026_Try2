"""
DEPRECATED — use ``scripts/download_data.py`` (the one authoritative path).

Reconciliation note: this legacy script pulled the WHOLE repo via
``snapshot_download`` with no per-file verification, no manifests and no
download gate. ``download_data.py`` supersedes it: it fetches only the exact
``{mode}/{split}`` ``.h5`` subsets, verifies each file (readability, shape,
labels, SHA-256) and writes machine-readable manifests, with
``--allow-download`` required before anything is fetched.

This module is kept as a backward-compatible shim that forwards to
``download_tsr_dataset``. New usage:

    python scripts/download_data.py --allow-download --output-dir D:/TSRD
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

try:  # imported as a package (tests / repo root on sys.path)
    from scripts.download_data import download_tsr_dataset
except ImportError:  # executed directly: python scripts/download_tsrd.py
    from download_data import download_tsr_dataset

DEPRECATION_MSG = (
    "\n[WARNING] scripts/download_tsrd.py is DEPRECATED: it was a whole-repo "
    "snapshot_download with no verification/manifests/gate.\n"
    "Use the one authoritative path instead:\n"
    "    python scripts/download_data.py --allow-download --output-dir D:/TSRD\n"
)


def main() -> None:
    print(DEPRECATION_MSG, file=sys.stderr)
    parser = argparse.ArgumentParser(
        description="DEPRECATED TSRD downloader — shim for scripts/download_data.py"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("TSRD_DATA_ROOT", "data"),
        help="Canonical dataset root (env TSRD_DATA_ROOT takes precedence; default data)",
    )
    parser.add_argument("--token", type=str, default=None, help="HF token (or set HF_TOKEN env)")
    parser.add_argument("--modes", nargs="+", default=["stare", "scan"], help="Modes: stare, scan")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"], help="Splits: train, validation, test")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="EXPLICIT confirmation to download (matches the authoritative script).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the exact file set without downloading (matches the authoritative script).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)
    logger.warning("scripts/download_tsrd.py is deprecated — see stderr for the canonical command.")

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
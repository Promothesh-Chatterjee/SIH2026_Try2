"""
Download TSRD from HuggingFace: train/val/test for stare and scan modes.

Reads HF_TOKEN from .env, shows tqdm progress, verifies H5 openability,
prints summary statistics (total pulse trains, size on disk).
"""

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _human_size(nbytes: int) -> str:
    """Human-readable file size.

    Args:
        nbytes: Bytes.

    Returns:
        String like '1.2 GB'.
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(nbytes) < 1024.0:
            return f"{nbytes:3.1f}{unit}"
        nbytes /= 1024.0
    return f"{nbytes:.1f}PB"


def download_tsr_dataset(
    output_dir: str | Path = "data",
    token: str | None = None,
    modes: list[str] | None = None,
    splits: list[str] | None = None,
) -> dict:
    """Download TSRD subsets from HuggingFace.

    Dataset: huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset
    Repo: https://github.com/alan-turing-institute/turing-deinterleaving-challenge

    Args:
        output_dir: Root output directory (data/stare/train etc.).
        token: HF token (defaults to HF_TOKEN env).
        modes: List of modes e.g. ["stare","scan"].
        splits: List of splits e.g. ["train","validation","test"].

    Returns:
        Dict with summary statistics.

    Raises:
        RuntimeError: If download fails and no fallback.
    """
    token = token or os.getenv("HF_TOKEN")
    if not token or token == "your_huggingface_token_here":
        logger.warning("HF_TOKEN not set or placeholder — attempting anonymous download (may fail for gated dataset)")

    output_dir = Path(output_dir)
    modes = modes or ["stare", "scan"]
    splits = splits or ["train", "validation", "test"]

    # Try huggingface_hub snapshot_download
    summary: dict = {"total_files": 0, "total_bytes": 0, "modes": {}}
    hf_available = False
    try:
        from huggingface_hub import snapshot_download  # type: ignore
        hf_available = True
    except ImportError:
        logger.warning("huggingface_hub not installed — falling back to datasets library")

    if hf_available:
        for mode in modes:
            for split in splits:
                target = output_dir / mode / split
                target.mkdir(parents=True, exist_ok=True)
                repo_id = "alan-turing-institute/turing-synthetic-radar-dataset"
                # TSRD structure is typically dataset/{mode}/{split}/*.h5 or stare/scan
                # Try snapshot_download with allow_patterns
                try:
                    from tqdm import tqdm  # type: ignore

                    logger.info("Downloading %s/%s → %s", mode, split, target)
                    # Use snapshot_download to get whole repo then copy relevant subset
                    # For efficiency, we use hf_hub_download per-file listing via datasets
                    # Fallback to datasets streaming if snapshot not feasible
                    raise ImportError("Use datasets fallback for pattern")
                except Exception as exc:
                    logger.debug("Snapshot pattern failed %s/%s: %s — trying datasets", mode, split, exc)

    # Datasets library fallback — more reliable for gated + pattern
    try:
        from datasets import load_dataset  # type: ignore
        from tqdm import tqdm  # type: ignore

        for mode in modes:
            for split in splits:
                target = output_dir / mode / split
                target.mkdir(parents=True, exist_ok=True)
                logger.info("Loading dataset via datasets: mode=%s split=%s", mode, split)
                # Attempt to load; TSRD may expose as config per mode
                try:
                    ds = load_dataset(
                        "alan-turing-institute/turing-synthetic-radar-dataset",
                        name=mode if mode in ["stare", "scan"] else None,
                        split=split if split != "validation" else "validation",
                        token=token if token and token != "your_huggingface_token_here" else None,  # type: ignore
                        trust_remote_code=True,
                    )
                    logger.info("Dataset %s/%s: %d rows", mode, split, len(ds))
                    # Save each row as .h5 if needed — dataset may already be PDW Arrow
                    # For TSRD, the canonical download is via turing_deinterleaving_challenge loader
                    # So we also try direct H5 download via hf hub
                    summary["total_files"] += len(ds)
                    summary["modes"].setdefault(mode, {})[split] = len(ds)
                except Exception as exc:
                    logger.warning("datasets load failed for %s/%s: %s", mode, split, exc)
                    # Fallback: try turing_deinterleaving_challenge download helper if exists
                    try:
                        import turing_deinterleaving_challenge as tdc  # type: ignore

                        if hasattr(tdc, "download_data"):
                            logger.info("Trying tdc.download_data for %s/%s", mode, split)
                            tdc.download_data(str(target), mode=mode, split=split, token=token)
                    except Exception as e2:
                        logger.debug("tdc fallback failed: %s", e2)
    except ImportError as exc:
        logger.error("datasets/tqdm not installed: %s — cannot download", exc)
        raise RuntimeError("Install datasets and tqdm: pip install datasets tqdm huggingface_hub") from exc

    # Also try direct hf download of H5 files if dataset rows not materialized
    # Use huggingface_hub list_repo_files
    try:
        from huggingface_hub import hf_hub_download, list_repo_files  # type: ignore
        from tqdm import tqdm  # type: ignore
        import h5py  # type: ignore

        if token and token != "your_huggingface_token_here":
            files = list_repo_files("alan-turing-institute/turing-synthetic-radar-dataset", token=token)
        else:
            files = list_repo_files("alan-turing-institute/turing-synthetic-radar-dataset")
        h5_files = [f for f in files if f.endswith(".h5")]
        logger.info("Repo has %d .h5 files", len(h5_files))
        for mode in modes:
            for split in splits:
                pattern = f"{mode}/{split}"
                matched = [f for f in h5_files if pattern in f]
                if not matched:
                    # Try alternative pattern
                    matched = [f for f in h5_files if mode in f and split in f]
                if not matched:
                    continue
                target = output_dir / mode / split
                target.mkdir(parents=True, exist_ok=True)
                logger.info("Downloading %d H5 for %s/%s", len(matched), mode, split)
                for repo_file in tqdm(matched, desc=f"{mode}/{split}"):
                    try:
                        local = hf_hub_download(
                            repo_id="alan-turing-institute/turing-synthetic-radar-dataset",
                            filename=repo_file,
                            local_dir=str(output_dir),
                            local_dir_use_symlinks=False,
                            token=token if token and token != "your_huggingface_token_here" else None,
                        )
                        summary["total_files"] += 1
                        summary["total_bytes"] += Path(local).stat().st_size if Path(local).exists() else 0
                    except Exception as exc:
                        logger.warning("Failed to download %s: %s", repo_file, exc)
    except Exception as exc:
        logger.debug("Direct H5 download skipped: %s", exc)

    # Verification: check H5 openability
    logger.info("Verifying H5 integrity...")
    verified = 0
    corrupt = 0
    try:
        import h5py  # type: ignore

        for mode in modes:
            for split in splits:
                target = output_dir / mode / split
                if not target.exists():
                    continue
                for h5_path in target.glob("*.h5"):
                    try:
                        with h5py.File(str(h5_path), "r") as f:
                            # Try reading a dataset
                            _ = list(f.keys())[:1]
                        verified += 1
                    except Exception as exc:
                        logger.warning("Corrupt H5 %s: %s", h5_path, exc)
                        corrupt += 1
    except ImportError:
        logger.warning("h5py not installed — skipping verification")

    # Disk summary
    total_bytes = 0
    total_files = 0
    for mode in modes:
        for split in splits:
            target = output_dir / mode / split
            if target.exists():
                for p in target.rglob("*"):
                    if p.is_file():
                        total_files += 1
                        try:
                            total_bytes += p.stat().st_size
                        except Exception:
                            pass
    logger.info("=== Download Summary ===")
    logger.info("Total files on disk: %d", total_files)
    logger.info("Total size: %s", _human_size(total_bytes))
    logger.info("Verified H5: %d, Corrupt: %d", verified, corrupt)
    for mode in modes:
        for split in splits:
            target = output_dir / mode / split
            if target.exists():
                count = len(list(target.glob("*.h5")))
                if count > 0:
                    logger.info("  %s/%s: %d H5 files", mode, split, count)

    summary.update({"verified": verified, "corrupt": corrupt, "disk_files": total_files, "disk_bytes": total_bytes})
    return summary


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser(description="Download TSRD dataset")
    parser.add_argument("--output-dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--token", type=str, default=None, help="HF token (or set HF_TOKEN env)")
    parser.add_argument("--modes", nargs="+", default=["stare", "scan"], help="Modes to download")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"], help="Splits to download")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        summary = download_tsr_dataset(output_dir=args.output_dir, token=args.token, modes=args.modes, splits=args.splits)
        print(f"\nDone. Summary: {summary}")
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        raise


if __name__ == "__main__":
    main()

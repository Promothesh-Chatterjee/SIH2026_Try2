#!/usr/bin/env python3
"""Provision the frozen Gate-25k baseline checkpoint, baseline evidence package,
and canonical TSRD test fixture from Azure Blob Storage.

Fail-closed in strict mode (default): any provisioning failure exits non-zero.
Best-effort mode (--best-effort): tolerates missing Azure credentials but still
fails on SHA mismatch or checkpoint corruption.

Downloads use atomic write: temp file → SHA verify → rename to canonical path.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Complete baseline evidence package + canonical TSRD fixture contract
PROVISION_TARGETS = [
    {
        "container": "smartscan-models",
        "blob_name": "scheduler_v2/checkpoint_gate_25000_frozen.pt",
        "local_path": "experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt",
        "expected_sha256": "7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0",
        "description": "Production Baseline Frozen Checkpoint",
    },
    {
        "container": "smartscan-models",
        "blob_name": "scheduler_v2/checkpoint_gate_25000_frozen.pt",
        "local_path": "experiments/checkpoints/scheduler_v2_operational_candidate/checkpoint_gate_25000_frozen.pt",
        "expected_sha256": "7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0",
        "description": "Candidate Mirror Frozen Checkpoint",
    },
    {
        "container": "smartscan-models",
        "blob_name": "scheduler_v2/SHA256SUMS",
        "local_path": "experiments/checkpoints/production_baseline/SHA256SUMS",
        "expected_sha256": "ce220b282b1e97a6a1cd06c429fff1254414dd772ca09be8ef296f51a4e0e621",
        "description": "Baseline Checksum Manifest (SHA256SUMS)",
    },
    {
        "container": "smartscan-models",
        "blob_name": "scheduler_v2/baseline_metadata.json",
        "local_path": "experiments/checkpoints/production_baseline/baseline_metadata.json",
        "expected_sha256": "9c0b10e45a43fd2d9fd1da2fb62c057e9c8a4fd16734e078d835cf7571c9d330",
        "description": "Baseline Architecture Metadata",
    },
    {
        "container": "smartscan-models",
        "blob_name": "scheduler_v2/benchmark_v2_baseline_gate25k.json",
        "local_path": "experiments/checkpoints/production_baseline/benchmark_v2_baseline_gate25k.json",
        "expected_sha256": "adf02d70699a3a8225dd67c823cc6679586086e2d86b4f6d4a9e078e2a0b58bd",
        "description": "Baseline Gate-25k Benchmark Metrics",
    },
    {
        "container": "smartscan-models",
        "blob_name": "scheduler_v2/benchmark_v2_multiseed_summary.json",
        "local_path": "experiments/checkpoints/production_baseline/benchmark_v2_multiseed_summary.json",
        "expected_sha256": "a44e70df97a52ddaa22be23eb81a293be1b3fc120e2595ca5b739c1d0cae81b1",
        "description": "Baseline Multi-Seed Summary",
    },
    {
        "container": "smartscan-models",
        "blob_name": "scheduler_v2/baseline_reservoir_5k.pkl",
        "local_path": "experiments/checkpoints/production_baseline/baseline_reservoir_5k.pkl",
        "expected_sha256": "edcef07b020563aefeac99fa3b03c2c6a474f07afdac7660e61e336523b8fe0c",
        "description": "Baseline Reservoir 5k Replay Buffer",
    },
    {
        "container": "tsrd-dataset",
        "blob_name": "val_stare/config_117.h5",
        "local_path": "tests/fixtures/canonical_tsrd/stare/val_stare/config_117.h5",
        "expected_sha256": "073724fbcd3aba8daaf94a68cbcd95ac1cf6f1aaeeab2b4b83df54e9a93dfe3f",
        "description": "Canonical TSRD STARE Validation Fixture (config_117)",
    },
]


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def atomic_download(blob_client, dest: Path, expected_sha: str) -> None:
    """Download blob to a temporary file, verify SHA, then atomically rename.

    If the SHA does not match, the temporary file is removed and no partial
    file ever appears at the canonical destination path.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path_str = tempfile.mkstemp(
        dir=str(dest.parent), suffix=".downloading"
    )
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "wb") as f:
            blob_client.download_blob().readinto(f)
        actual = sha256_file(tmp_path)
        if actual != expected_sha:
            print(f"[FAIL] SHA-256 mismatch after download: expected {expected_sha}, got {actual}")
            tmp_path.unlink(missing_ok=True)
            raise ValueError(
                f"Downloaded file SHA {actual} != expected {expected_sha}"
            )
        # Atomic rename (same filesystem)
        shutil.move(str(tmp_path), str(dest))
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Fail-closed: any provisioning failure exits non-zero (default).",
    )
    mode.add_argument(
        "--best-effort",
        action="store_true",
        help="Tolerate missing Azure credentials (exit 0). Still fail on corruption.",
    )
    args = parser.parse_args()
    strict = not args.best_effort

    conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")

    if not conn_str:
        if strict:
            print(
                "[FAIL] AZURE_STORAGE_CONNECTION_STRING not set (strict mode)."
            )
            return 1
        else:
            print(
                "[SKIP] AZURE_STORAGE_CONNECTION_STRING not set (best-effort mode)."
            )
            return 0

    try:
        from azure.storage.blob import BlobServiceClient
        from azure.core.exceptions import ResourceNotFoundError
    except ImportError:
        print("[FAIL] azure-storage-blob package not installed.")
        return 1

    try:
        client = BlobServiceClient.from_connection_string(conn_str)
    except Exception as e:
        print(f"[FAIL] Could not connect to Azure Blob Storage: {e}")
        return 1

    for target in PROVISION_TARGETS:
        local = Path(target["local_path"])
        expected_sha = target["expected_sha256"]
        container = target["container"]
        blob_name = target["blob_name"]
        desc = target["description"]

        # If already present with correct SHA, skip download
        if local.exists():
            actual = sha256_file(local)
            if actual == expected_sha:
                print(f"[OK] Already present and verified: {local} ({desc})")
                continue
            else:
                print(
                    f"[WARN] {local} SHA mismatch ({actual[:16]}...) — re-downloading."
                )
                local.unlink()  # Remove corrupt/wrong file

        print(f"[INFO] Downloading [{container}] {blob_name} -> {local} ({desc})")
        try:
            blob_client = client.get_blob_client(
                container=container, blob=blob_name
            )
            atomic_download(blob_client, local, expected_sha)
            size_kb = local.stat().st_size / 1024
            if size_kb >= 1024:
                print(f"[OK] Downloaded & verified: {local} ({size_kb / 1024:.1f} MB)")
            else:
                print(f"[OK] Downloaded & verified: {local} ({size_kb:.1f} KB)")
        except ResourceNotFoundError:
            print(f"[FAIL] Blob not found: {container}/{blob_name}")
            return 1
        except ValueError:
            return 1
        except Exception as e:
            print(f"[FAIL] Download failed: {e}")
            return 1

    # Verify all files exist and match SHA
    for target in PROVISION_TARGETS:
        local = Path(target["local_path"])
        expected_sha = target["expected_sha256"]
        if not local.exists():
            print(f"[FAIL] Expected file not found after provisioning: {local}")
            return 1
        actual = sha256_file(local)
        if actual != expected_sha:
            print(f"[FAIL] Final SHA check failed for {local}: {actual}")
            return 1

    # Full baseline package manifest verification
    try:
        from scripts.verify_baseline_gate import verify_sha256sums
        base_dir = Path("experiments/checkpoints/production_baseline")
        if not verify_sha256sums(base_dir):
            print("[FAIL] Baseline package SHA256SUMS manifest verification failed!")
            return 1
        print("[OK] Baseline package SHA256SUMS verified across all components.")
    except Exception as e:
        print(f"[FAIL] Baseline manifest verification error: {e}")
        return 1

    print("\n[OK] Complete baseline evidence package and canonical TSRD fixture provisioned and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

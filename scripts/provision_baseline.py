#!/usr/bin/env python3
"""Provision the frozen Gate-25k baseline checkpoint from Azure Blob Storage.

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

EXPECTED_SHA256 = "7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0"
CONTAINER = "smartscan-models"
BLOB_NAME = "scheduler_v2/checkpoint_gate_25000_frozen.pt"

LOCAL_TARGETS = [
    "experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt",
    "experiments/checkpoints/scheduler_v2_operational_candidate/checkpoint_gate_25000_frozen.pt",
]


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def atomic_download(blob_client, dest: Path) -> None:
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
        if actual != EXPECTED_SHA256:
            print(f"[FAIL] SHA-256 mismatch after download: {actual}")
            tmp_path.unlink(missing_ok=True)
            raise ValueError(
                f"Downloaded file SHA {actual} != expected {EXPECTED_SHA256}"
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

    # List blobs for diagnostics
    try:
        container_client = client.get_container_client(CONTAINER)
        blobs = list(container_client.list_blobs())
        print(f"[INFO] Found {len(blobs)} blob(s) in '{CONTAINER}':")
        for b in blobs:
            print(f"       {b.name} ({b.size / 1024 / 1024:.1f} MB)")
        if not blobs:
            print("[FAIL] Container is empty — checkpoint not uploaded.")
            return 1
    except Exception as e:
        print(f"[FAIL] Could not list blobs: {e}")
        return 1

    for local_rel in LOCAL_TARGETS:
        local = Path(local_rel)

        # If already present with correct SHA, skip download
        if local.exists():
            actual = sha256_file(local)
            if actual == EXPECTED_SHA256:
                print(f"[OK] Already present and verified: {local}")
                continue
            else:
                print(
                    f"[WARN] {local} SHA mismatch ({actual[:16]}...) — re-downloading."
                )
                local.unlink()  # Remove corrupt/wrong file

        print(f"[INFO] Downloading {BLOB_NAME} -> {local}")
        try:
            blob_client = client.get_blob_client(
                container=CONTAINER, blob=BLOB_NAME
            )
            atomic_download(blob_client, local)
            size_mb = local.stat().st_size / 1024 / 1024
            print(f"[OK] Downloaded and verified: {local} ({size_mb:.1f} MB)")
        except ResourceNotFoundError:
            print(f"[FAIL] Blob not found: {CONTAINER}/{BLOB_NAME}")
            return 1
        except ValueError:
            # SHA mismatch already printed by atomic_download
            return 1
        except Exception as e:
            print(f"[FAIL] Download failed: {e}")
            return 1

    # Final verification: both targets must exist with correct SHA
    for local_rel in LOCAL_TARGETS:
        local = Path(local_rel)
        if not local.exists():
            print(f"[FAIL] Expected checkpoint not found: {local}")
            return 1
        actual = sha256_file(local)
        if actual != EXPECTED_SHA256:
            print(f"[FAIL] Final SHA check failed for {local}: {actual}")
            return 1

    print("[OK] All baseline checkpoint copies provisioned and verified.")
    print(f"[OK] SHA-256: {EXPECTED_SHA256}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

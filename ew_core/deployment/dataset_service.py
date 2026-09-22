"""TSRD Dataset service — abstracts local D:/TSRD vs Azure Blob Storage.

In development: reads from local path set in TSRD_DATA_ROOT env var.
In Azure production: reads from AZURE_BLOB_ACCOUNT/AZURE_BLOB_CONTAINER or /mnt/tsrd mount.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

TSRD_DATA_ROOT = os.environ.get("TSRD_DATA_ROOT", "D:/TSRD")
AZURE_BLOB_ACCOUNT = os.environ.get("AZURE_BLOB_ACCOUNT", "")
AZURE_BLOB_CONTAINER = os.environ.get("AZURE_BLOB_CONTAINER", "tsrd-dataset")


def get_tsrd_root() -> str:
    """Return the TSRD root path — local or mounted Azure Blob."""
    # 1. In Azure AKS with Blob CSI Driver, the blob is mounted at /mnt/tsrd
    azure_mount = Path("/mnt/tsrd")
    if azure_mount.exists():
        return str(azure_mount)

    # 2. Local development configured via environment variable
    local_env = os.environ.get("TSRD_DATA_ROOT", TSRD_DATA_ROOT)
    local = Path(local_env)
    if local.exists():
        return str(local)

    # 3. Check fallback repository local data path
    fallback_data = Path("data")
    if fallback_data.exists():
        return str(fallback_data)

    raise RuntimeError(
        f"TSRD dataset not found. Set TSRD_DATA_ROOT env var (local) "
        f"or mount Azure Blob at /mnt/tsrd (Azure AKS)."
    )


def list_scenarios(subset: str = "val") -> List[str]:
    """List available scenario IDs in the TSRD dataset."""
    try:
        root = Path(get_tsrd_root())
    except RuntimeError:
        return []

    # Check stare/{subset}_stare, {subset}_stare, {subset}, or root
    candidates = [
        root / "stare" / f"{subset}_stare",
        root / f"{subset}_stare",
        root / subset,
        root,
    ]
    for c in candidates:
        if c.exists():
            files = list(c.glob("config_*.h5")) + list(c.glob("config_*.json"))
            if files:
                return sorted(list({p.stem for p in files}))
    return []


async def download_from_blob(blob_uri: str, target_dir: str = "experiments/checkpoints/downloaded") -> str:
    """Download a checkpoint/dataset file from Azure Blob Storage given az://container/blob or https URL."""
    target_p = Path(target_dir)
    target_p.mkdir(parents=True, exist_ok=True)

    clean_path = blob_uri.replace("az://", "").replace("https://", "")
    parts = clean_path.split("/", 1)
    blob_name = parts[1] if len(parts) > 1 else parts[0]
    filename = Path(blob_name).name
    dest_path = target_p / filename

    conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
    if conn_str:
        try:
            from azure.storage.blob import BlobServiceClient
            container = parts[0] if len(parts) > 1 else AZURE_BLOB_CONTAINER
            blob_service = BlobServiceClient.from_connection_string(conn_str)
            blob_client = blob_service.get_blob_client(container=container, blob=blob_name)
            data = blob_client.download_blob().readall()
            with open(dest_path, "wb") as f:
                f.write(data)
            logger.info("Downloaded %s from Azure Blob to %s", blob_uri, dest_path)
            return str(dest_path)
        except Exception as exc:
            logger.error("Failed to download from Azure Blob %s: %s", blob_uri, exc)
            raise RuntimeError(f"Azure Blob download failed: {exc}")

    # Fallback if local file exists matching target
    if dest_path.exists():
        return str(dest_path)
    raise RuntimeError(f"Cannot download {blob_uri}: AZURE_STORAGE_CONNECTION_STRING not configured")

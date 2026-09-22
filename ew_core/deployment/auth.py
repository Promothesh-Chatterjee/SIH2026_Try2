"""API key authentication middleware for the SmartScan EW API."""

from __future__ import annotations

import logging
import os
import secrets
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

API_KEY_HEADER = APIKeyHeader(name="X-SmartScan-API-Key", auto_error=False)


def get_valid_api_keys() -> set[str]:
    """Retrieve currently valid API keys dynamically from environment variable."""
    return {
        k.strip()
        for k in os.environ.get("SMARTSCAN_API_KEYS", "").split(",")
        if k.strip()
    }


def require_api_key(api_key: str | None = Security(API_KEY_HEADER)) -> str:
    """FastAPI Dependency: validate API key from request header or raise 401."""
    valid_keys = get_valid_api_keys()
    if not valid_keys:
        # Development mode: if no keys configured, allow all (log warning)
        logger.warning("No API keys configured in SMARTSCAN_API_KEYS — running in open dev mode")
        return "dev-open"

    if not api_key or api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Provide X-SmartScan-API-Key header.",
        )
    return api_key


def generate_api_key() -> str:
    """Generate a cryptographically secure API key."""
    return f"sk-smartscan-{secrets.token_hex(32)}"

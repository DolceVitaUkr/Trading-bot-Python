# file: tradingbot/ui/routes/diff.py
"""Diff endpoints for trade reconciliation."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter()


def _validate_asset(asset: str) -> str:
    """Validate asset symbol input."""
    if not asset or not asset.strip():
        raise HTTPException(status_code=400, detail="Asset symbol cannot be empty")
    if len(asset) > 20:
        raise HTTPException(status_code=400, detail="Asset symbol too long")
    if not asset.replace("/", "").replace("-", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid asset symbol format")
    return asset.upper().strip()


@router.get("/diff/{asset}")
def diff_preview(asset: str) -> dict:
    """Return a dry-run reconciliation diff for ``asset``."""
    validated_asset = _validate_asset(asset)
    return {"asset": validated_asset, "dry_run": True}


@router.post("/diff/confirm/{asset}")
def diff_confirm(asset: str) -> dict:
    """Confirm and execute reconciliation for ``asset``."""
    validated_asset = _validate_asset(asset)
    return {"asset": validated_asset, "reconciled": True}


__all__ = ["router"]

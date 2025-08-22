"""Diff endpoints for trade reconciliation."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/diff/{asset}")
def diff_preview(asset: str) -> dict:
    """Return a dry-run reconciliation diff for ``asset``."""
    return {"asset": asset, "dry_run": True}


@router.post("/diff/confirm/{asset}")
def diff_confirm(asset: str) -> dict:
    """Confirm and execute reconciliation for ``asset``."""
    return {"asset": asset, "reconciled": True}


__all__ = ["router"]

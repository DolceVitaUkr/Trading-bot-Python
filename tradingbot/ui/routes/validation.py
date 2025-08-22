"""Validation report endpoints."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter()

LOG_DIR = Path(__file__).resolve().parents[3] / "logs" / "validation"


def get_latest_report(strategy_id: str) -> dict:
    """Return the latest validation report for ``strategy_id``.

    The report is expected at ``logs/validation/{strategy_id}/summary.json``.
    """
    path = LOG_DIR / strategy_id / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"report not found for {strategy_id}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@router.get("/validation/{strategy_id}")
def validation(strategy_id: str) -> dict:
    """Serve the validation report for ``strategy_id``."""
    try:
        return get_latest_report(strategy_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="report not found") from exc


__all__ = ["router", "get_latest_report"]

# file: core/validationmanager.py
"""Compatibility wrapper for validation_manager module with new helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from .validation_manager import ValidationManager as _ValidationManager


class ValidationManager(_ValidationManager):
    """Expose convenience methods using underscore-free names."""

    def savereport(self, report: Dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)

    def latestreport(self, strategyid: str) -> Dict[str, Any] | None:
        return super().latest_report(strategyid)

    def eligibleforlive(self, strategyid: str, gates: Dict[str, Any] | None = None) -> bool:
        passed, _ = super().eligible_for_live(strategyid)
        if not gates:
            return passed
        if not passed:
            return False
        report = self.latestreport(strategyid) or {}
        metrics = report.get("metrics", {})
        trades = report.get("trades", 0)
        if trades < gates.get("trades", 0):
            return False
        for key, threshold in gates.items():
            if key in metrics and metrics[key] < threshold:
                return False
        return True


__all__ = ["ValidationManager"]

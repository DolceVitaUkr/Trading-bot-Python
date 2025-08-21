# file: core/driftmonitor.py
"""Drift monitoring utilities for feature drift detection.

This lightweight implementation compares the mean of each feature against a
baseline and reports a z-score.  It also integrates with the :class:`Notifier`
so that alerts can be emitted when drift exceeds a configurable threshold.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from .notifier import Notifier


class DriftMonitor:
    """Detect simple feature drift based on mean deviation."""

    def __init__(self, notifier: Notifier | None = None, threshold: float = 3.0) -> None:
        self.notifier = notifier or Notifier()
        self.threshold = threshold

    def checkdrift(self, features: pd.DataFrame, baseline: pd.DataFrame) -> Dict[str, Any]:
        """Return drift statistics comparing ``features`` to ``baseline``.

        The result maps each common column to a dictionary with the mean
        difference and z-score relative to the baseline's standard deviation.
        ``max_zscore`` contains the maximum absolute z-score across all
        features for quick threshold checks.
        """
        report: Dict[str, Any] = {}
        columns = sorted(set(features.columns) & set(baseline.columns))
        for column in columns:
            f = features[column].dropna()
            b = baseline[column].dropna()
            if b.empty or f.empty:
                continue
            mean_diff = f.mean() - b.mean()
            std = b.std() or 1.0
            z = mean_diff / std
            report[column] = {"mean_diff": float(mean_diff), "zscore": float(z)}
        report["max_zscore"] = max((abs(v["zscore"]) for v in report.values()), default=0.0)
        return report

    def alertifdrift(self, report: Dict[str, Any]) -> None:
        """Send an alert if the drift report exceeds the configured threshold."""
        max_z = report.get("max_zscore", 0.0)
        if max_z > self.threshold:
            self.notifier.send(f"drift detected: z={max_z:.2f}")


__all__ = ["DriftMonitor"]

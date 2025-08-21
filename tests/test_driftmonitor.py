import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from tradingbot.core.driftmonitor import DriftMonitor


def test_checkdrift_reports_zscore():
    baseline = pd.DataFrame({"a": [0, 0, 0, 0], "b": [1, 1, 1, 1]})
    features = pd.DataFrame({"a": [0, 0, 1, 1], "b": [1, 2, 1, 2]})
    dm = DriftMonitor(threshold=2)
    report = dm.checkdrift(features, baseline)
    assert "a" in report and "b" in report
    assert report["a"]["zscore"] != 0
    assert report["max_zscore"] >= max(abs(report["a"]["zscore"]), abs(report["b"]["zscore"]))


def test_alertifdrift_triggers_notifier():
    baseline = pd.DataFrame({"x": [0, 0, 0, 0]})
    features = pd.DataFrame({"x": [10, 10, 10, 10]})
    dm = DriftMonitor(threshold=1)
    report = dm.checkdrift(features, baseline)
    dm.alertifdrift(report)
    assert dm.notifier.messages  # should contain alert


def test_alertifdrift_no_alert_below_threshold():
    baseline = pd.DataFrame({"x": [0, 0, 0, 0]})
    features = pd.DataFrame({"x": [0.1, -0.1, 0.2, -0.2]})
    dm = DriftMonitor(threshold=5)
    report = dm.checkdrift(features, baseline)
    dm.alertifdrift(report)
    assert not dm.notifier.messages

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parent.parent))

from tradingbot.ui.app import app, diff_manager, validator


def test_diff_preview_returns_proposals_without_execution():
    client = TestClient(app)
    asset = "BTC"
    actions = ["buy", "sell"]
    diff_manager.set_proposed_actions(asset, actions)

    resp = client.get(f"/diff/{asset}")
    assert resp.status_code == 200
    assert resp.json() == {"asset": asset, "actions": actions}
    assert diff_manager.applied_actions(asset) == []


def test_diff_confirm_applies_actions():
    client = TestClient(app)
    asset = "ETH"
    actions = ["close"]
    diff_manager.set_proposed_actions(asset, actions)

    resp = client.post(f"/diff/confirm/{asset}")
    assert resp.status_code == 200
    assert resp.json() == {"asset": asset, "actions": actions}
    assert diff_manager.applied_actions(asset) == actions
    assert diff_manager.proposed_actions(asset) == []


def test_validation_endpoint_serves_latest_report(tmp_path):
    client = TestClient(app)
    strategy = "alpha"
    report = {"passed": True, "reasons": []}

    strat_dir = tmp_path / strategy
    strat_dir.mkdir(parents=True)
    (strat_dir / "summary.json").write_text(json.dumps(report))
    validator.log_dir = tmp_path

    resp = client.get(f"/validation/{strategy}")
    assert resp.status_code == 200
    assert resp.json() == report

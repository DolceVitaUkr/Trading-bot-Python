import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from tradingbot.core.configmanager import ConfigManager
from tradingbot.core.riskmanager import RiskManager, OrderProposal
from tradingbot.core.runtimecontroller import RuntimeController


def test_config_env_substitution(monkeypatch):
    monkeypatch.setenv("BYBIT_API_KEY", "TESTKEY")
    cm = ConfigManager(Path(__file__).resolve().parent.parent / "tradingbot" / "config")
    assert cm.get("api_keys.bybit.key") == "TESTKEY"


def test_risk_manager_stop_loss_limit():
    rm = RiskManager(max_stop_loss_pct=0.15)
    ok, _ = rm.validateorder(OrderProposal(price=100, stop_loss=90))
    assert ok
    ok, reason = rm.validateorder(OrderProposal(price=100, stop_loss=50))
    assert not ok
    assert "exceeds" in reason.lower()


def test_runtime_validation_and_kill_switch(tmp_path):
    rc = RuntimeController(state_path=tmp_path / "runtime.json", validator=lambda a: a == "BTCUSDT")
    with pytest.raises(ValueError):
        rc.enablelive("ETHUSDT")
    rc.enablelive("BTCUSDT")
    assert rc.getstate()["assets"]["BTCUSDT"]["live"] is True
    rc.setglobalkill(True)
    state = rc.getstate()
    assert state["global"]["kill_switch"] is True
    assert state["assets"]["BTCUSDT"]["close_only"] is True

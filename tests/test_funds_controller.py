import pytest
from modules.Funds_Controller import FundsController

def test_initial_state():
    """Tests the default initial state of the controller."""
    fc = FundsController()
    assert not fc.allow_bot_funds
    assert fc.asset_enabled["SPOT"]
    assert not fc.asset_enabled["PERP"]
    assert fc.max_pair_allocation_pct == 0.10

def test_initial_state_override():
    """Tests initializing the controller with a custom state."""
    initial_state = {
        "allow_bot_funds": True,
        "asset_enabled": {"SPOT": False, "PERP": True},
        "max_pair_allocation_pct": 0.25,
    }
    fc = FundsController(initial_state=initial_state)
    assert fc.allow_bot_funds
    assert not fc.asset_enabled["SPOT"]
    assert fc.asset_enabled["PERP"]
    assert fc.max_pair_allocation_pct == 0.25

def test_is_allowed():
    """Tests the core is_allowed logic."""
    fc = FundsController()
    # Bot funds not allowed
    assert not fc.is_allowed("SPOT", "BTC/USDT")

    fc.set_allow_bot_funds(True)
    # Bot funds allowed, SPOT enabled
    assert fc.is_allowed("SPOT", "BTC/USDT")
    # PERP is not enabled by default
    assert not fc.is_allowed("PERP", "ETH/USDT:USDT")

    fc.set_asset_enabled("PERP", True)
    # PERP now enabled
    assert fc.is_allowed("PERP", "ETH/USDT:USDT")

    fc.set_asset_enabled("SPOT", False)
    # SPOT now disabled
    assert not fc.is_allowed("SPOT", "BTC/USDT")

def test_pair_cap_pct():
    """Tests the pair_cap_pct getter."""
    fc = FundsController()
    assert fc.pair_cap_pct() == 0.10
    fc.set_max_pair_allocation_pct(0.15)
    assert fc.pair_cap_pct() == 0.15

def test_snapshot():
    """Tests the state snapshot functionality."""
    fc = FundsController()
    fc.set_allow_bot_funds(True)
    fc.set_asset_enabled("FOREX", True)
    snap = fc.snapshot()

    assert snap["allow_bot_funds"]
    assert snap["asset_enabled"]["FOREX"]
    assert "timestamp_utc" in snap

def test_state_modification_logging(caplog):
    """Tests that state changes are logged."""
    fc = FundsController()
    with caplog.at_level("INFO"):
        fc.set_allow_bot_funds(True, source="Test")
        assert "Parameter 'allow_bot_funds' changed from 'False' to 'True'" in caplog.text
        caplog.clear()

        fc.set_asset_enabled("PERP", True, source="API")
        assert "Parameter 'asset_enabled.PERP' changed from 'False' to 'True'" in caplog.text
        caplog.clear()

        fc.set_max_pair_allocation_pct(0.20)
        assert "Parameter 'max_pair_allocation_pct' changed from '0.1' to '0.2'" in caplog.text

def test_invalid_pair_cap_pct():
    """Tests that invalid values for max_pair_allocation_pct are rejected."""
    fc = FundsController()
    fc.set_max_pair_allocation_pct(0.005) # Too low
    assert fc.pair_cap_pct() == 0.10
    fc.set_max_pair_allocation_pct(0.51) # Too high
    assert fc.pair_cap_pct() == 0.10

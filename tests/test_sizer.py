import pytest
import json
import os
from modules.Sizer import Sizer

@pytest.fixture(scope="module")
def sizer_policy():
    return {
      "global": {
        "equity_threshold_usd": 20000,
        "fixed_trade_usd": 10,
        "max_risk_pct": 0.005, # 0.5%
        "max_risk_pct_good_setup": 0.0075, # 0.75%
        "atr_mult_sl": 1.2,
        "min_stop_distance_pct": 0.0015, # 0.15%
        "good_setup_score_min": 0.80
      },
      "leverage_tiers": [
        {"equity_max": 10000,  "SCALP": 1.0, "TREND": 1.0},
        {"equity_max": 25000,  "SCALP": 2.0, "TREND": 1.5},
        {"equity_max": 50000,  "SCALP": 3.0, "TREND": 2.0}
      ],
      "asset_caps": {
        "SPOT":   {"max_leverage": 1.0},
        "PERP":   {"max_leverage": 3.0},
        "FOREX":  {"max_leverage": 5.0}
      }
    }

@pytest.fixture
def sizer(sizer_policy):
    return Sizer(policy=sizer_policy)

def test_sizer_from_json(sizer_policy):
    """Tests loading the policy from a JSON file."""
    path = "tests/temp_sizing_policy.json"
    with open(path, 'w') as f:
        json.dump(sizer_policy, f)

    sizer = Sizer.from_json(path)
    assert sizer.global_policy['fixed_trade_usd'] == 10
    os.remove(path)

def test_fixed_sizing_below_threshold(sizer):
    """Tests that fixed sizing is used when equity is below the threshold."""
    proposal = sizer.propose(
        equity=19999, asset_class="SPOT", mode="TREND",
        atr=200, price=50000, pair_cap_pct=0.10
    )
    assert proposal['size_usd'] == 10

def test_risk_percent_sizing_above_threshold(sizer):
    """Tests basic %-risk sizing when equity is above the threshold."""
    equity = 30000
    price = 50000
    atr = 250 # ATR in price terms
    # sl_pct = max(1.2 * 250 / 50000, 0.0015) = max(0.006, 0.0015) = 0.006
    # risk_per_trade = 30000 * 0.005 = 150
    # size_usd = 150 / 0.006 = 25000
    proposal = sizer.propose(
        equity=equity, asset_class="SPOT", mode="TREND",
        atr=atr, price=price, pair_cap_pct=1.0 # Use high cap to not interfere
    )
    assert proposal['size_usd'] == pytest.approx(25000)

def test_good_setup_boosts_risk(sizer):
    """Tests that a good setup uses a higher risk percentage."""
    equity = 30000
    price = 50000
    atr = 250
    # sl_pct is still 0.006
    # risk_per_trade = 30000 * 0.0075 = 225
    # size_usd = 225 / 0.006 = 37500
    proposal = sizer.propose(
        equity=equity, asset_class="SPOT", mode="TREND",
        atr=atr, price=price, pair_cap_pct=1.0, # Use high cap to not interfere
        good_setup=True, signal_score=0.85
    )
    # The uncapped size would be 37500, but it's capped by equity * pair_cap_pct (30000 * 1.0 = 30000)
    assert proposal['size_usd'] == pytest.approx(30000)

def test_pair_allocation_cap(sizer):
    """Tests that the final size is capped by the pair allocation percentage."""
    equity = 30000
    price = 50000
    atr = 250
    pair_cap_pct = 0.10 # 10% of 30k = 3000
    # Uncapped size would be 25000, but should be capped at 3000
    proposal = sizer.propose(
        equity=equity, asset_class="SPOT", mode="TREND",
        atr=atr, price=price, pair_cap_pct=pair_cap_pct
    )
    assert proposal['size_usd'] == pytest.approx(3000)

def test_leverage_tiering(sizer):
    """Tests that leverage is correctly selected from tiers."""
    # Equity 9k -> 1.0x
    prop1 = sizer.propose(equity=9000, asset_class="PERP", mode="SCALP", atr=1, price=100, pair_cap_pct=0.1)
    assert prop1['leverage'] == 1.0

    # Equity 22k -> 2.0x for SCALP
    prop2 = sizer.propose(equity=22000, asset_class="PERP", mode="SCALP", atr=1, price=100, pair_cap_pct=0.1)
    assert prop2['leverage'] == 2.0

    # Equity 22k -> 1.5x for TREND
    prop3 = sizer.propose(equity=22000, asset_class="PERP", mode="TREND", atr=1, price=100, pair_cap_pct=0.1)
    assert prop3['leverage'] == 1.5

def test_leverage_asset_cap(sizer):
    """Tests that leverage is capped by the asset's max leverage."""
    # Equity 40k -> SCALP would be 3.0x, but PERP cap is 3.0x. So 3.0x
    prop1 = sizer.propose(equity=40000, asset_class="PERP", mode="SCALP", atr=1, price=100, pair_cap_pct=0.1)
    assert prop1['leverage'] == 3.0

    # Equity 40k -> SCALP would be 3.0x, but SPOT cap is 1.0x
    prop2 = sizer.propose(equity=40000, asset_class="SPOT", mode="SCALP", atr=1, price=100, pair_cap_pct=0.1)
    assert prop2['leverage'] == 1.0

def test_min_stop_distance(sizer):
    """Tests that the minimum stop distance is enforced."""
    equity = 30000
    price = 50000
    atr = 10 # ATR is very small, so min_stop_distance_pct should dominate
    # atr_sl_pct = 1.2 * 10 / 50000 = 0.00024
    # min_sl_pct = 0.0015
    # sl_pct is 0.0015
    sl_distance = 0.0015 * price

    proposal = sizer.propose(
        equity=equity, asset_class="SPOT", mode="TREND",
        atr=atr, price=price, pair_cap_pct=0.10
    )
    assert proposal['sl_distance'] == pytest.approx(sl_distance)

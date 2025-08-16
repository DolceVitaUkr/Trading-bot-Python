import pytest
import json
import os
from modules.Sizer import Sizer

@pytest.fixture(scope="module")
def sizer_policy():
    # This policy is now mostly for fallbacks and global settings,
    # as the core logic is phased inside the Sizer.
    return {
      "global": {
        "max_risk_pct": 0.01, # 1% for phase 4
        "max_risk_pct_good_setup": 0.02, # 2% for phase 4 good setups
        "atr_mult_sl": 1.2,
        "min_stop_distance_pct": 0.0015,
        "good_setup_score_min": 0.80
      },
      "leverage_tiers": [
        {"equity_max": 50000, "DEFAULT": 1.0},
        {"equity_max": 999999, "DEFAULT": 2.0}
      ],
      "asset_caps": {
        "SPOT": {"max_leverage": 1.0},
        "PERP": {"max_leverage": 5.0}
      }
    }

@pytest.fixture
def sizer(sizer_policy):
    return Sizer(policy=sizer_policy)

# --- Tests for the new Phased Scaling Logic ---

def test_phase_1_fixed_sizing(sizer):
    """Phase 1 (equity <= 1000): Fixed $10 trade."""
    proposal = sizer.propose(
        equity=900, asset_class="SPOT", mode="TREND",
        atr=200, price=50000, pair_cap_pct=0.10, signal_score=0.7
    )
    assert proposal['size_usd'] == 10.0

def test_phase_2_direct_equity_sizing(sizer):
    """Phase 2 (1k < equity <= 5k): 0.5%-1.0% of equity."""
    # Test at mid-range signal score
    proposal = sizer.propose(
        equity=4000, asset_class="SPOT", mode="TREND",
        atr=200, price=50000, pair_cap_pct=0.10, signal_score=0.5
    )
    # Expected size_pct = 0.005 + (0.5 * 0.005) = 0.0075
    # Expected size_usd = 4000 * 0.0075 = 30
    assert proposal['size_usd'] == pytest.approx(30.0)

    # Test at max signal score
    proposal_max = sizer.propose(
        equity=4000, asset_class="SPOT", mode="TREND",
        atr=200, price=50000, pair_cap_pct=0.10, signal_score=1.0
    )
    # Expected size_pct = 0.005 + (1.0 * 0.005) = 0.01
    # Expected size_usd = 4000 * 0.01 = 40
    assert proposal_max['size_usd'] == pytest.approx(40.0)

def test_phase_3_risk_percentage_sizing(sizer):
    """Phase 3 (5k < equity <= 20k): Risk scales from 0.5% to 5%."""
    equity = 15000
    price = 50000
    atr = 250
    # sl_pct = max(1.2 * 250 / 50000, 0.0015) = 0.006

    # Test at low signal score (risk should be close to 0.5%)
    proposal_low = sizer.propose(
        equity=equity, asset_class="SPOT", mode="TREND",
        atr=atr, price=price, pair_cap_pct=100.0, signal_score=0.1 # High cap
    )
    # risk_pct = 0.005 + (0.1 * (0.05 - 0.005)) = 0.005 + 0.0045 = 0.0095
    # risk_usd = 15000 * 0.0095 = 142.5
    # size_usd = 142.5 / 0.006 = 23750
    assert proposal_low['size_usd'] == pytest.approx(23750)

    # Test at high signal score (risk should be close to 5%)
    proposal_high = sizer.propose(
        equity=equity, asset_class="SPOT", mode="TREND",
        atr=atr, price=price, pair_cap_pct=100.0, signal_score=0.9 # High cap
    )
    # risk_pct = 0.005 + (0.9 * 0.045) = 0.005 + 0.0405 = 0.0455
    # risk_usd = 15000 * 0.0455 = 682.5
    # size_usd = 682.5 / 0.006 = 113750
    assert proposal_high['size_usd'] == pytest.approx(113750)


def test_phase_4_advanced_risk_sizing(sizer):
    """Phase 4 (equity > 20k): Uses risk % from policy."""
    equity = 30000
    price = 50000
    atr = 250
    # sl_pct = 0.006
    # risk_pct from policy = 0.01 (base)
    # risk_usd = 30000 * 0.01 = 300
    # size_usd = 300 / 0.006 = 50000
    proposal = sizer.propose(
        equity=equity, asset_class="SPOT", mode="TREND",
        atr=atr, price=price, pair_cap_pct=100.0 # High cap
    )
    assert proposal['size_usd'] == pytest.approx(50000)

def test_phase_4_good_setup_boost_and_cap(sizer):
    """Phase 4 (equity > 20k): Good setup boosts risk and is capped at 10% of equity."""
    equity = 100000
    price = 50000
    atr = 250
    # sl_pct = 0.006
    # good_setup risk_pct from policy = 0.02
    # risk_usd = 100000 * 0.02 = 2000
    # size_usd (uncapped) = 2000 / 0.006 = 333333.33
    # 10% equity cap = 100000 * 0.10 = 10000
    proposal = sizer.propose(
        equity=equity, asset_class="SPOT", mode="TREND",
        atr=atr, price=price, pair_cap_pct=1.0, # High pair cap
        good_setup=True, signal_score=0.9
    )
    # The size should be clamped to the 10% equity cap
    assert proposal['size_usd'] == pytest.approx(10000)

def test_global_pair_allocation_cap_still_applies(sizer):
    """Tests that the global pair allocation cap is always respected."""
    # Test in Phase 2
    proposal_p2 = sizer.propose(
        equity=4000, asset_class="SPOT", mode="TREND",
        atr=200, price=50000, pair_cap_pct=0.05, signal_score=1.0
    )
    # Uncapped size = 4000 * 0.01 = 40
    # Capped size = 4000 * 0.05 = 200. The uncapped is smaller.
    # Let's try a smaller cap
    proposal_p2_capped = sizer.propose(
        equity=4000, asset_class="SPOT", mode="TREND",
        atr=200, price=50000, pair_cap_pct=0.005, signal_score=1.0
    )
    # Uncapped size = 40. Capped size = 4000 * 0.005 = 20
    assert proposal_p2_capped['size_usd'] == pytest.approx(20)

    # Test in Phase 4
    proposal_p4 = sizer.propose(
        equity=100000, asset_class="SPOT", mode="TREND",
        atr=250, price=50000, pair_cap_pct=0.05, # 5% cap
        good_setup=True, signal_score=0.9
    )
    # Good setup capped at 10% equity (10000), but pair cap is 5% (5000)
    assert proposal_p4['size_usd'] == pytest.approx(5000)

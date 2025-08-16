import pytest
from modules.Validation_Manager import ValidationManager

@pytest.fixture
def validation_manager():
    """Provides a fresh ValidationManager instance for each test."""
    config = {
        "min_trades_for_approval": 500,
        "min_sharpe_ratio": 1.0,
        "min_out_of_sample_sharpe": 0.5
    }
    # We pass an empty config and then set it, to avoid running the initial validation for every test.
    # We will test the initial validation separately.
    vm = ValidationManager(config={})
    vm.config = config
    vm.min_trades = config["min_trades_for_approval"]
    vm.min_sharpe = config["min_sharpe_ratio"]
    vm.min_oos_sharpe = config["min_out_of_sample_sharpe"]
    vm.approved_strategies = {}
    vm.rejected_strategies = {}
    return vm

# --- Test Cases ---

def test_approval_pass(validation_manager):
    """Test a strategy that should pass all validation criteria."""
    strategy_id = "GoodStrategy"
    asset_class = "SPOT"
    bt_results = {"trades": 600, "sharpe_ratio": 1.5}
    wfa_results = {"out_of_sample_sharpe": 1.2}

    validation_manager.run_validation_for_strategy(strategy_id, asset_class, bt_results, wfa_results)

    assert validation_manager.is_strategy_approved(strategy_id, asset_class) is True
    assert not validation_manager.rejected_strategies.get(asset_class)

def test_rejection_insufficient_trades(validation_manager):
    """Test rejection due to not enough trades."""
    strategy_id = "NotEnoughTrades"
    asset_class = "SPOT"
    bt_results = {"trades": 499, "sharpe_ratio": 1.5}
    wfa_results = {"out_of_sample_sharpe": 1.2}

    validation_manager.run_validation_for_strategy(strategy_id, asset_class, bt_results, wfa_results)

    assert validation_manager.is_strategy_approved(strategy_id, asset_class) is False
    rejection_log = validation_manager.rejected_strategies[asset_class][0]
    assert "Insufficient trades" in rejection_log["reason"]

def test_rejection_low_sharpe_ratio(validation_manager):
    """Test rejection due to a low Sharpe ratio."""
    strategy_id = "LowSharpe"
    asset_class = "SPOT"
    bt_results = {"trades": 600, "sharpe_ratio": 0.9}
    wfa_results = {"out_of_sample_sharpe": 1.2}

    validation_manager.run_validation_for_strategy(strategy_id, asset_class, bt_results, wfa_results)

    assert validation_manager.is_strategy_approved(strategy_id, asset_class) is False
    rejection_log = validation_manager.rejected_strategies[asset_class][0]
    assert "Sharpe ratio too low" in rejection_log["reason"]

def test_rejection_low_out_of_sample_sharpe(validation_manager):
    """Test rejection due to a low out-of-sample Sharpe ratio."""
    strategy_id = "LowOOSSharpe"
    asset_class = "PERP"
    bt_results = {"trades": 600, "sharpe_ratio": 1.5}
    wfa_results = {"out_of_sample_sharpe": 0.4}

    validation_manager.run_validation_for_strategy(strategy_id, asset_class, bt_results, wfa_results)

    assert validation_manager.is_strategy_approved(strategy_id, asset_class) is False
    rejection_log = validation_manager.rejected_strategies[asset_class][0]
    assert "Out-of-sample Sharpe ratio too low" in rejection_log["reason"]

def test_initial_validation_at_startup():
    """Test that the manager is pre-populated correctly on initialization."""
    config = {
        "min_trades_for_approval": 500,
        "min_sharpe_ratio": 1.0,
        "min_out_of_sample_sharpe": 0.5
    }
    vm = ValidationManager(config)

    # Check approved strategy
    assert vm.is_strategy_approved("TrendFollowStrategy", "SPOT") is True
    assert vm.is_strategy_approved("MeanReversionStrategy", "PERP") is True

    # Check rejected strategy
    assert vm.is_strategy_approved("MeanReversionStrategy", "SPOT") is False
    assert "SPOT" in vm.rejected_strategies
    spot_rejections = [r for r in vm.rejected_strategies["SPOT"] if r["strategy_id"] == "MeanReversionStrategy"]
    assert len(spot_rejections) == 1
    assert "Insufficient trades" in spot_rejections[0]["reason"]

def test_asset_class_differentiation(validation_manager):
    """Test that approval for one asset class doesn't imply approval for another."""
    strategy_id = "AssetSpecificStrategy"

    # Approve for SPOT
    bt_results_spot = {"trades": 700, "sharpe_ratio": 1.8}
    wfa_results_spot = {"out_of_sample_sharpe": 1.1}
    validation_manager.run_validation_for_strategy(strategy_id, "SPOT", bt_results_spot, wfa_results_spot)

    # Reject for PERP
    bt_results_perp = {"trades": 700, "sharpe_ratio": 0.8} # Low sharpe
    wfa_results_perp = {"out_of_sample_sharpe": 1.1}
    validation_manager.run_validation_for_strategy(strategy_id, "PERP", bt_results_perp, wfa_results_perp)

    assert validation_manager.is_strategy_approved(strategy_id, "SPOT") is True
    assert validation_manager.is_strategy_approved(strategy_id, "PERP") is False
    assert "PERP" in validation_manager.rejected_strategies

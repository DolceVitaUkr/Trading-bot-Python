import pytest
from modules.Portfolio_Manager import PortfolioManager

class MockWalletSync:
    def __init__(self, is_live=False):
        self.is_live = is_live
        self.balances = {}

    def get_equity(self, asset):
        return self.balances.get(asset, 0.0)

    def set_simulation_balance(self, asset, balance):
        if not self.is_live:
            self.balances[asset] = balance

@pytest.fixture
def mock_wallet_sync_sim():
    return MockWalletSync(is_live=False)

@pytest.fixture
def mock_wallet_sync_live():
    ws = MockWalletSync(is_live=True)
    ws.balances = {"SPOT": 5000, "PERP": 2000}
    return ws

def test_pm_initialization_fixed():
    """Tests PortfolioManager initialization with fixed USD allocations."""
    allocations = {"SPOT": 10000, "PERP": 5000}
    pm = PortfolioManager(allocations=allocations)
    assert pm.ledgers["SPOT"]["total"] == 10000
    assert pm.available_budget("SPOT") == 10000
    assert pm.ledgers["PERP"]["available"] == 5000

def test_pm_initialization_percentage():
    """Tests PortfolioManager initialization with percentage-based allocations."""
    allocations = {"SPOT": 0.8, "PERP": 0.2}
    pm = PortfolioManager(allocations=allocations, is_percentage=True, total_capital=20000)
    assert pm.ledgers["SPOT"]["total"] == 16000
    assert pm.available_budget("PERP") == 4000

def test_pm_initialization_informs_wallet_sync(mock_wallet_sync_sim):
    """Tests that PM sets initial simulation balances in WalletSync."""
    allocations = {"SPOT": 10000}
    _ = PortfolioManager(allocations=allocations, wallet_sync=mock_wallet_sync_sim)
    assert mock_wallet_sync_sim.get_equity("SPOT") == 10000

def test_available_budget_simulation(mock_wallet_sync_sim):
    """Tests available_budget in simulation mode."""
    pm = PortfolioManager(allocations={"SPOT": 10000}, wallet_sync=mock_wallet_sync_sim)
    assert pm.available_budget("SPOT") == 10000

def test_available_budget_live_capping(mock_wallet_sync_live):
    """Tests that live wallet equity caps the available budget."""
    # Ledger has 10k, but live wallet only has 5k
    pm = PortfolioManager(allocations={"SPOT": 10000}, wallet_sync=mock_wallet_sync_live)
    assert pm.available_budget("SPOT") == 5000

    # Ledger has 1k, live wallet has 5k -> should return 1k
    pm2 = PortfolioManager(allocations={"SPOT": 1000}, wallet_sync=mock_wallet_sync_live)
    assert pm2.available_budget("SPOT") == 1000

def test_reserve_and_release():
    """Tests the full reservation and release cycle."""
    pm = PortfolioManager(allocations={"SPOT": 1000})
    assert pm.available_budget("SPOT") == 1000

    # Successful reservation
    rid = pm.reserve("SPOT", 200)
    assert rid is not None
    assert pm.available_budget("SPOT") == 800
    assert len(pm.reservations) == 1

    # Failed reservation (insufficient funds)
    rid2 = pm.reserve("SPOT", 801)
    assert rid2 is None
    assert pm.available_budget("SPOT") == 800

    # Release reservation
    pm.release(rid)
    assert pm.available_budget("SPOT") == 1000
    assert len(pm.reservations) == 0

def test_book_trade_and_release_cycle():
    """Tests the cycle of reserve -> book_trade -> release."""
    pm = PortfolioManager(allocations={"SPOT": 1000})

    # 1. Reserve capital
    rid = pm.reserve("SPOT", 100)
    assert pm.available_budget("SPOT") == 900
    assert pm.ledgers["SPOT"]["total"] == 1000

    # 2. Book a profitable trade
    pm.book_trade(asset="SPOT", pnl_net=20, fees=2)
    # Total equity should increase by PnL
    assert pm.ledgers["SPOT"]["total"] == 1020
    # Available budget also increases by PnL
    assert pm.ledgers["SPOT"]["available"] == 920
    assert pm.ledgers["SPOT"]["realized_pnl"] == 20
    assert pm.ledgers["SPOT"]["fees"] == 2

    # 3. Release the original reserved amount
    pm.release(rid)
    # Available budget should now equal total equity
    assert pm.available_budget("SPOT") == 1020
    assert pm.ledgers["SPOT"]["total"] == 1020

def test_book_trade_updates_wallet_sync_sim(mock_wallet_sync_sim):
    """Tests that booking a trade updates WalletSync's balance in sim mode."""
    pm = PortfolioManager(allocations={"SPOT": 1000}, wallet_sync=mock_wallet_sync_sim)
    assert mock_wallet_sync_sim.get_equity("SPOT") == 1000

    rid = pm.reserve("SPOT", 100)
    pm.book_trade("SPOT", pnl_net=-50, fees=5)
    pm.release(rid)

    assert pm.ledgers["SPOT"]["total"] == 950
    assert mock_wallet_sync_sim.get_equity("SPOT") == 950

from modules.Wallet_Sync import WalletSync

class MockExchangeAdapter:
    def __init__(self, balance, is_simulation=True, api_should_fail=False):
        self.balance = balance
        self.is_simulation = is_simulation
        self.api_should_fail = api_should_fail

    def get_balance(self):
        if self.api_should_fail:
            raise ConnectionError("API call failed")
        return self.balance

def test_wallet_sync_initialization():
    """Tests that WalletSync initializes correctly."""
    adapters = {
        "SPOT": MockExchangeAdapter(10000),
        "PERP": MockExchangeAdapter(5000, is_simulation=False)
    }
    ws = WalletSync(exchange_adapters=adapters)
    assert ws.is_live
    assert ws.get_equity("SPOT") == 0.0 # Initial balances are 0 until sync
    assert ws.get_equity("PERP") == 0.0

def test_sync_simulation_mode():
    """Tests that sync does nothing in simulation mode."""
    adapters = {
        "SPOT": MockExchangeAdapter(10000, is_simulation=True),
        "PERP": MockExchangeAdapter(5000, is_simulation=True)
    }
    ws = WalletSync(exchange_adapters=adapters)
    assert not ws.is_live

    ws.set_simulation_balance("SPOT", 12000)
    assert ws.get_equity("SPOT") == 12000

    synced_balances = ws.sync()
    assert synced_balances["SPOT"] == 12000 # Should return last known, not live balance
    assert ws.get_equity("SPOT") == 12000


def test_sync_live_mode_success():
    """Tests a successful sync in live mode."""
    adapters = {
        "SPOT": MockExchangeAdapter(10000, is_simulation=False),
        "PERP": MockExchangeAdapter(5000, is_simulation=False)
    }
    ws = WalletSync(exchange_adapters=adapters)
    synced_balances = ws.sync()

    assert len(synced_balances) == 2
    assert synced_balances["SPOT"] == 10000
    assert synced_balances["PERP"] == 5000
    assert ws.get_equity("SPOT") == 10000
    assert ws.get_equity("PERP") == 5000

def test_sync_live_mode_api_failure():
    """Tests that sync uses the last known balance when an API call fails."""
    spot_adapter = MockExchangeAdapter(10000, is_simulation=False)
    perp_adapter = MockExchangeAdapter(5000, is_simulation=False, api_should_fail=True)

    adapters = {"SPOT": spot_adapter, "PERP": perp_adapter}
    ws = WalletSync(exchange_adapters=adapters)

    # First sync is successful for SPOT, fails for PERP (initial balance is 0)
    synced_balances_1 = ws.sync()
    assert synced_balances_1["SPOT"] == 10000
    assert synced_balances_1["PERP"] == 0.0
    assert ws.get_equity("SPOT") == 10000
    assert ws.get_equity("PERP") == 0.0

    # PERP API is now fixed, SPOT fails
    perp_adapter.api_should_fail = False
    spot_adapter.api_should_fail = True

    synced_balances_2 = ws.sync()
    # SPOT should use last known balance
    assert synced_balances_2["SPOT"] == 10000
    # PERP should be updated
    assert synced_balances_2["PERP"] == 5000
    assert ws.get_equity("SPOT") == 10000
    assert ws.get_equity("PERP") == 5000

def test_get_equity_unknown_asset():
    """Tests get_equity for an asset not in the adapters."""
    adapters = {"SPOT": MockExchangeAdapter(10000)}
    ws = WalletSync(exchange_adapters=adapters)
    assert ws.get_equity("FOREX") == 0.0

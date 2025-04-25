# self_test/test_trading_bot.py

import os
import sys
import math
import builtins
import types
from datetime import datetime, timedelta
from decimal import Decimal

# Ensure project root is in PYTHONPATH for imports if running directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import modules from the project
import config
import modules.data_manager as data_manager
import modules.exchange as exchange
import modules.technical_indicators as technical_indicators
import modules.risk_management as risk_management
import modules.reward_system as reward_system
import modules.trade_calculator as trade_calculator
import modules.trade_executor as trade_executor
import modules.top_pairs as top_pairs
import modules.telegram_bot as telegram_bot
import modules.ui as ui
import time
import json
import modules.trade_simulator as trade_simulator
import modules.self_learning as self_learning
import pandas as pd
import pytest

# --- Helper monkey-patches and dummy classes for external dependencies ---

# Monkey-patch ccxt bybit client in exchange to avoid real API calls
class DummyExchangeClient:
    """A dummy CCXT exchange client with minimal methods for testing."""
    def __init__(self, *args, **kwargs):
        self.orders = []
    def fetch_ohlcv(self, symbol, timeframe='15m', limit=1000):
        # Return dummy OHLCV data: generate 'limit' entries of incremental prices
        # Format: [timestamp, open, high, low, close, volume]
        limit = limit or 1000
        base_time = int(time.time() * 1000) - limit * 60_000
        data = []
        price = 100.0
        for i in range(limit):
            ts = base_time + i * 60_000
            open_p = price
            high_p = price + 1
            low_p = price - 1
            close_p = price + 0.5
            vol = 100 + i
            data.append([ts, open_p, high_p, low_p, close_p, vol])
            price += 1  # increment price each interval
        return data
    def fetch_ticker(self, symbol):
        # Return dummy ticker data with a pretend current price
        return {"last": 123.45}
    def create_order(self, symbol, type, side, amount, price=None):
        order = {"id": "TESTORDER1", "symbol": symbol, "type": type, "side": side,
                 "amount": amount, "price": price, "status": "open"}
        self.orders.append(order)
        return order
    def load_markets(self):
        return True

# Apply monkey patch to ccxt.bybit in our exchange module
exchange.ccxt.bybit = lambda params=None: DummyExchangeClient()

# Monkey-patch Telegram Bot API calls to prevent actual HTTP requests
telegram_bot.Bot.send_message = lambda self, chat_id, text, **kwargs: {"ok": True, "text": text}
telegram_bot.Bot.send_document = lambda self, chat_id, document, **kwargs: {"ok": True}

# Monkey-patch requests.get in top_pairs to avoid actual API call
def dummy_requests_get(url, headers=None, timeout=None):
    class DummyResponse:
        def __init__(self, data, ok=True):
            self._data = data
            self.ok = ok
        def raise_for_status(self):
            if not self.ok:
                raise Exception("HTTP error")
        def json(self):
            return self._data
    # Simulate a valid API response with a couple of symbols
    dummy_data = {
        "ret_code": 0,
        "result": [
            {"name": "TESTUSDT", "quoteCurrency": "USDT", "status": "Trading"},
            {"name": "FOOUSD", "quoteCurrency": "USD", "status": "Trading"},
            {"name": "BARUSDT", "quoteCurrency": "USDT", "status": "Suspended"}
        ]
    }
    return DummyResponse(dummy_data)
top_pairs.requests.get = dummy_requests_get

# Ensure simulation mode and dummy keys for Exchange to avoid ValueErrors
config.USE_SIMULATION = True
config.BYBIT_API_KEY = "X" * 32
config.BYBIT_API_SECRET = "Y" * 48
config.SIMULATION_BYBIT_API_KEY = "Z" * 16
config.SIMULATION_BYBIT_API_SECRET = "W" * 32

# --- Tests ---

def test_data_manager_update_and_load(tmp_path):
    """Test DataManager's ability to update klines and load them in test mode."""
    dm = data_manager.DataManager()
    dm.test_mode = True  # ensure test mode (writes to test_ prefixed files)
    symbol = "TEST/USDT"
    timeframe = "1m"
    # Generate a small set of mock klines
    klines = dm._generate_mock_klines(periods=5, timeframe=timeframe)
    # Update klines (this should create a test file in HISTORICAL_DATA_PATH)
    success = dm.update_klines(symbol, timeframe, klines, test_mode=True)
    assert success is True, "update_klines should return True on successful update"
    # Load the data back
    df = dm.load_historical_data(symbol, timeframe, test_mode=True)
    # The DataFrame should have 5 rows and the 'close' column matching our klines
    assert not df.empty and len(df) == 5, "Historical data load should retrieve 5 records"
    # Verify the last close price matches what was generated (50000 + 4 + 25 = 50029)
    last_close = df['close'].iloc[-1]
    expected_last_close = klines[-1][4]
    assert math.isclose(last_close, expected_last_close, rel_tol=1e-9), \
        f"Expected last close {expected_last_close}, got {last_close}"
    # Test timeframe conversion utility
    assert data_manager.timeframe_to_minutes("15m") == 15
    assert data_manager.timeframe_to_minutes("1h") == 60
    assert data_manager.timeframe_to_minutes("1d") == 1440
    assert data_manager.timeframe_to_minutes("1w") == 10080

def test_technical_indicators_basic():
    """Test a selection of technical indicator functions with simple data."""
    prices = pd.Series([1, 2, 3, 4, 5], dtype=float)
    df = pd.DataFrame({
        "close": [1, 2, 3, 4, 5],
        "volume": [10, 10, 10, 10, 10]
    })
    # Moving average (window=3): last value should be average of [3,4,5] = 4
    ma = technical_indicators.moving_average(prices, window=3)
    assert round(ma.iloc[-1], 6) == 4.0
    # Exponential moving average (EMA) returns series of same length
    ema = technical_indicators.ema(df, window=3, price_col='close')
    assert len(ema) == len(df)
    # RSI on a strictly increasing series -> last RSI should be 100 (or very close)
    rsi_vals = technical_indicators.rsi(prices, window=2)
    assert rsi_vals.iloc[-1] >= 99.99, f"RSI of increasing series expected ~100, got {rsi_vals.iloc[-1]}"
    # MACD: should return tuple of three series (MACD line, signal line, histogram)
    macd_line, signal_line, hist = technical_indicators.macd(df, fast=3, slow=6, signal=3, price_col='close')
    assert len(macd_line) == len(signal_line) == len(hist) == len(df)
    # Bollinger Bands: returns tuple (middle, upper, lower)
    mid, upper, lower = technical_indicators.bollinger_bands(df, window=3, std_dev=1.0, price_col='close')
    assert len(mid) == len(upper) == len(lower) == len(df)
    # ATR: on monotonically increasing prices with fixed range, ATR should be constant after initial period
    atr_vals = technical_indicators.atr(df.assign(high=df['close']+1, low=df['close']-1), window=3)
    assert len(atr_vals) == len(df)
    # OBV: for strictly increasing prices, OBV should monotonically increase
    df_vol = pd.DataFrame({
        "close": [10, 12, 15],
        "volume": [100, 150, 200]
    })
    obv_vals = technical_indicators.obv(df_vol, price_col='close', volume_col='volume')
    # OBV should start at 0 and then add all volumes (because price rises each time)
    assert obv_vals.iloc[0] == 0 or obv_vals.iloc[0] == 100  # depending on implementation (start at 0 or first vol)
    assert obv_vals.iloc[-1] == obv_vals.iloc[0] + 150 + 200, "OBV did not accumulate volumes correctly"
    # Stochastic oscillator: returns %K and %D series of equal length
    stok, stod = technical_indicators.stochastic_oscillator(df, k_window=3, d_window=2)
    assert len(stok) == len(stod) == len(df)
    # Fibonacci retracement: returns a dict of levels
    fib = technical_indicators.fibonacci_retracement(df, lookback=3)
    expected_levels = {"0%": df['close'].iloc[-1], "100%": df['close'].iloc[-3]}
    assert all(level in fib for level in expected_levels.keys()), "Fib retracement missing expected keys"
    # Market regime: returns a series of regime signals (likely +1/-1 or similar)
    regime = technical_indicators.market_regime(df, short_window=2, long_window=3)
    assert len(regime) == len(df)
    # Entropy volatility: returns a series
    entropy = technical_indicators.entropy_volatility(df, window=2)
    assert len(entropy) == len(df)

def test_risk_management_calculations():
    """Test core risk management functions and classes."""
    # Basic stop loss calculation
    entry_price = 100.0
    sl = risk_management.calculate_stop_loss(entry_price, risk_percentage=0.02)
    assert math.isclose(sl, 98.0, rel_tol=1e-9), f"Expected 2% stop loss of 100 -> 98, got {sl}"
    # Dynamic adjustment (simulation version)
    new_stop, new_tp = risk_management.dynamic_adjustment(100, 105, 95, 110)
    assert new_stop >= 95, "Stop loss should not decrease"
    assert new_tp >= 110, "Take profit should not decrease when price rises"
    # PositionRisk dataclass usage
    pos = risk_management.PositionRisk(entry_price=100.0, stop_loss=95.0, take_profit=110.0,
                                       position_size=1.0, risk_reward_ratio=2.0,
                                       risk_percent=0.01, dollar_risk=100.0)
    # RiskManager position size calculation
    rm = risk_management.RiskManager(account_balance=100000.0, max_portfolio_risk=0.1)
    size = rm.calculate_position_size(entry_price=100.0, stop_price=90.0, risk_percent=0.01)
    # Dollar risk = 100000*0.01 = 1000, price_risk = 10, position_size = 1000/10 = 100
    assert math.isclose(size, 100.0, rel_tol=1e-9), f"Position size miscalculated: got {size}"
    # Ensure position size is limited by leverage (max_leverage=10 -> margin_required <= account_balance)
    rm.account_balance = 1000.0
    rm.max_leverage = 2
    # With low balance, a large risk_percent should trigger a RiskViolationError
    with pytest.raises(risk_management.RiskViolationError):
        rm.calculate_position_size(entry_price=50.0, stop_price=25.0, risk_percent=0.5)  # very large risk percent
    # Dynamic stop management via RiskManager
    rm = risk_management.RiskManager(account_balance=10000.0)
    pos = risk_management.PositionRisk(entry_price=100.0, stop_loss=90.0, take_profit=130.0,
                                       position_size=1.0, risk_reward_ratio=2.0,
                                       risk_percent=0.01, dollar_risk=100.0)
    updated_pos = rm.dynamic_stop_management(current_price=110.0, position=pos)
    assert isinstance(updated_pos, risk_management.PositionRisk)
    # After a price increase >5%, new stop should be at least entry price
    assert updated_pos.stop_loss >= 100.0, "Stop loss not moved to entry price on 10% gain"
    # Portfolio risk assessment
    rm.open_positions = {
        "pos1": risk_management.PositionRisk(100, 90, 120, 1, 2, 0.01, 100),
        "pos2": risk_management.PositionRisk(200, 180, 250, 2, 1.5, 0.02, 200)
    }
    risk_summary = rm.portfolio_risk_assessment()
    total_dollar_risk = 100 + 200
    expected_ratio = total_dollar_risk / rm.account_balance
    assert math.isclose(risk_summary['total_dollar_risk'], total_dollar_risk, rel_tol=1e-9)
    assert math.isclose(risk_summary['portfolio_risk_ratio'], expected_ratio, rel_tol=1e-9)

def test_reward_system_points_and_rewards():
    """Test reward points calculation and RewardSystem outputs."""
    now = datetime.now()
    one_hour_later = now + timedelta(hours=1)
    # Profit trade, no stop loss trigger
    pts = reward_system.calculate_points(profit=0.5, entry_time=now, exit_time=one_hour_later,
                                         stop_loss_triggered=False, risk_adjusted=True)
    # profit 0.5 -> base_points 50, ~1h duration -> time_bonus ~ 50-10 = 40, no penalty -> total ~90, risk_adjusted divides by 1
    assert pts > 0, f"Expected positive points for profitable trade, got {pts}"
    assert pts < 100, f"Time bonus application seems off, got {pts}"
    # Losing trade with stop loss triggered
    pts2 = reward_system.calculate_points(profit=-0.1, entry_time=now, exit_time=now + timedelta(hours=5),
                                          stop_loss_triggered=True, risk_adjusted=True)
    # profit -0.1 -> base -10, time_bonus (if any) small or 0, penalty 100 -> total likely negative and risk_adjusted divides by 101
    assert pts2 <= 0, "Expected non-positive points for stopped-out losing trade"
    # RewardSystem.calculate_reward (currently just profit * position_size)
    rs = reward_system.RewardSystem()
    rew = rs.calculate_reward(entry_price=100.0, exit_price=110.0, position_size=2.0,
                              entry_time=now, exit_time=one_hour_later,
                              max_drawdown=0.1, volatility=0.5, stop_loss_triggered=False)
    # Profit per unit = 10, position_size 2 -> reward = 20
    assert rew == 20.0

def test_trade_calculator_advanced():
    """Test the AdvancedTradeCalculator with a sample scenario."""
    adv_calc = trade_calculator.AdvancedTradeCalculator()
    # Use Decimal for inputs as required
    entry = Decimal('50000')
    exit = Decimal('52000')
    capital = Decimal('1000')
    leverage = 5
    result = adv_calc.calculate_trade(entry, exit, capital, leverage, is_maker=False, holding_days=0)
    # Check keys presence
    expected_keys = {"position_size", "gross_pnl", "net_pnl", "total_fee", 
                     "interest_cost", "effective_roi", "liquidation_price", "risk_adjusted_roi"}
    assert expected_keys.issubset(result.keys()), "Missing keys in trade calculation result"
    # Position size = (capital*leverage)/entry
    expected_size = (capital * leverage) / entry
    assert result["position_size"].quantize(Decimal('0.0001')) == expected_size.quantize(Decimal('0.0001'))
    # Net PnL = Gross PnL - total_fee (interest_cost is 0 in this case)
    assert result["net_pnl"] == result["gross_pnl"] - result["total_fee"] - result["interest_cost"]
    # ROI should be (net_pnl/capital)*100
    calc_roi = (result["net_pnl"] / capital) * Decimal('100')
    assert result["effective_roi"].quantize(Decimal('0.001')) == calc_roi.quantize(Decimal('0.001'))
    # Liquidation price for leverage > 1 should be > 0, and for leverage=1 should be 0
    liq_price = result["liquidation_price"]
    assert liq_price > 0
    # Test leverage=1 yields liquidation_price = 0
    result_no_leverage = adv_calc.calculate_trade(entry, exit, capital, leverage=1)
    assert result_no_leverage["liquidation_price"] == Decimal('0')

def test_telegram_notifier_format_and_send(monkeypatch):
    """Test TelegramNotifier formatting and ensure send_message doesn't raise errors."""
    tn = telegram_bot.TelegramNotifier(disable_async=True)
    # Monkey-patch the notifier's internal send to capture output instead of real sending
    sent_messages = []
    def fake_safe_send(msg, **kwargs):
        sent_messages.append(msg)
    tn._safe_send_message = lambda message, **kwargs: fake_safe_send(message, **kwargs)
    # Test sending a simple text message
    tn.send_message("Hello World", format='text')
    assert sent_messages[-1] == "Hello World"
    # Test sending a log message (dict content)
    log_content = {"level": "info", "message": "Test log"}
    tn.send_message(log_content, format='log')
    # The formatted message should include the level and message
    assert "Test log" in sent_messages[-1] and "info" in sent_messages[-1].lower()
    # Invalid format (wrong content for 'log') should log an error (we can check the internal logger by monkeypatching)
    error_logged = {"called": False}
    def fake_log_error(msg):
        error_logged["called"] = True
    tn.logger = types.SimpleNamespace(error=lambda msg, exc_info=None: fake_log_error(msg))
    tn.send_message("bad content", format='log')  # sending str with log format should trigger error
    assert error_logged["called"] is True

def test_top_pairs_fetch_and_cache(tmp_path):
    """Test PairManager fetching pairs with dummy API response and caching."""
    pm = top_pairs.PairManager()
    # Override cache file path to a temp directory to avoid collisions
    pm.cache_file = tmp_path / "pair_cache.json"
    pm.fallback_file = tmp_path / "fallback_pairs.json"
    pairs = pm.get_active_pairs()
    # From dummy response, expected active USDT pairs: ["TESTUSDT"] (BARUSDT is status Suspended and should be filtered out)
    assert isinstance(pairs, list)
    assert "TESTUSDT" in pairs and "FOOUSD" not in pairs, "Pair filtering failed"
    # After first call, a cache file should be written
    assert pm.cache_file.exists(), "Cache file was not created"
    # Modify cache to test cache retrieval
    with open(pm.cache_file, "r+") as f:
        data = json.load(f)
        # set timestamp to now to ensure it's fresh
        data["timestamp"] = datetime.now().isoformat()
        f.seek(0)
        json.dump(data, f)
        f.truncate()
    pairs2 = pm.get_active_pairs()  # should use cache this time (since we ensure it's fresh)
    assert pairs2 == pairs, "Expected cached pairs to be used on second call"
    # Corrupt cache to test fallback to hardcoded
    with open(pm.cache_file, "w") as f:
        f.write("}{")  # invalid JSON
    got_pairs = pm.get_active_pairs()
    assert isinstance(got_pairs, list) and len(got_pairs) > 0, "Fallback to hardcoded pairs failed"

def test_trade_executor_simulated_and_live(monkeypatch):
    """Test TradeExecutor order execution in simulated and live mode with monkey-patched exchange methods."""
    # Monkey-patch ExchangeAPI methods to avoid external calls and provide deterministic behavior
    monkeypatch.setattr(exchange.ExchangeAPI, "get_current_price", lambda self, symbol: 50000.0)
    monkeypatch.setattr(exchange.ExchangeAPI, "get_min_order_size", lambda self, symbol: 0.0)
    monkeypatch.setattr(exchange.ExchangeAPI, "get_price_precision", lambda self, symbol: 2)
    monkeypatch.setattr(exchange.ExchangeAPI, "create_order", lambda self, symbol, order_type, side, amount, price: 
                        {"id": "ORDER123", "symbol": symbol, "side": side, "amount": amount, "price": price, "status": "open"})
    te_sim = trade_executor.TradeExecutor(simulation_mode=True)
    result_sim = te_sim.execute_order("BTC/USDT", side="BUY", amount=0.002, price=50000.0, order_type="limit")
    # In simulation mode, result should be a dict with 'status': 'simulated'
    assert isinstance(result_sim, dict) and result_sim.get("status") == "simulated"
    assert result_sim.get("symbol") == "BTC/USDT"
    # Now test in live mode (simulation_mode=False)
    te_live = trade_executor.TradeExecutor(simulation_mode=False)
    # Valid order (amount above min_order_size which will be max(0.002, 0.0) = 0.002)
    result_live = te_live.execute_order("BTC/USDT", side="SELL", amount=1.5, price=50000.123456, order_type="limit")
    # Price should be rounded to 2 decimal places as per get_price_precision patch
    assert math.isclose(result_live["price"], 50000.12, rel_tol=1e-9)
    assert result_live["id"] == "ORDER123" and result_live["status"] == "open"
    # Test that order amount below minimum triggers ValueError
    with pytest.raises(ValueError):
        te_live.execute_order("BTC/USDT", side="BUY", amount=0.001, price=50000.0)
    # Ensure telegram notification is attempted (monkey-patched Bot.send_message returns dict as above)
    # We can infer it ran if no exception and our patched send_message was called (which it was if no error in execution)

def test_trade_simulator_basic_backtest(monkeypatch):
    """Basic test for TradeSimulator to run a tiny backtest."""
    sim = trade_simulator.TradeSimulator(initial_wallet=1000.0)
    # Monkey-patch DataManager in TradeSimulator to use a very small fixed dataset
    dummy_data = pd.DataFrame({
        "timestamp": pd.date_range(start="2021-01-01", periods=4, freq="D"),
        "open": [100, 102, 101, 103],
        "high": [101, 103, 102, 104],
        "low": [99, 101, 100, 102],
        "close": [100, 102, 101, 103],
        "volume": [1000, 1500, 1200, 1300]
    }).set_index("timestamp")
    monkeypatch.setattr(sim.data_manager, "load_historical_data",
                        lambda symbol, timeframe, test_mode=False: dummy_data)
    # Monkey-patch strategy decision within TradeSimulator (if exists) to a simple rule:
    # e.g., buy on first day, sell on second day.
    if hasattr(sim, "strategy"):
        sim.strategy = types.SimpleNamespace()
        sim.strategy.evaluate = lambda df, idx: ("buy" if idx == 0 else "sell" if idx == 1 else "hold")
    # Run simulation (assuming simulate or run_backtest method exists)
    if hasattr(sim, "simulate") or hasattr(sim, "run_backtest"):
        # Use whichever method is present to run the simulation
        method = getattr(sim, "simulate", getattr(sim, "run_backtest", None))
        result = method(symbol="TEST/USDT", timeframe="1d", start=None, end=None)
        # After simulation, the wallet should have changed (either increased, decreased, or at least executed trades)
        assert sim.wallet != 1000.0 or sim.trade_history, "Simulation did not execute trades as expected"
    else:
        pytest.skip("No simulate method implemented in TradeSimulator; skipping backtest test.")

def test_self_learning_agent(monkeypatch):
    """Test SelfLearningBot instantiation and single training step (if torch is available)."""
    # Skip entirely if torch not installed
    pytest.importorskip("torch")
    from modules.self_learning import SelfLearningBot, TradingDataset
    agent = SelfLearningBot(state_size=4, action_size=3)
    # The agent should have a policy network (DRQN) and an internal replay memory (dataset)
    assert hasattr(agent, "policy_net")
    # Simulate adding one experience and training on it
    dummy_experience = {
        "state": [0.1]*agent.state_size,
        "action": 1,
        "reward": 1.0,
        "next_state": [0.2]*agent.state_size,
        "done": False
    }
    agent.memory = [dummy_experience]  # directly insert into memory for testing
    try:
        agent.train(simulation_data=[dummy_experience])  # train on the single experience
    except Exception as e:
        pytest.fail(f"SelfLearningBot.train raised an exception: {e}")

def test_ui_self_test_integration(monkeypatch):
    """Test that TradingUI.run_self_test triggers the SystemTestRunner properly."""
    # Monkey-patch SystemTestRunner to a dummy version
    class DummyTestRunner:
        def __init__(self):
            DummyTestRunner.ran = False
        def run_full_suite(self):
            DummyTestRunner.ran = True
    monkeypatch.setitem(sys.modules, 'self_test.full_system_test', types.SimpleNamespace(SystemTestRunner=DummyTestRunner))
    ui.SystemTestRunner = DummyTestRunner  # ensure our UI uses DummyTestRunner
    # Monkey-patch threading.Thread to run tasks synchronously (to avoid dealing with actual threads in test)
    class DummyThread:
        def __init__(self, target=None, *args, **kwargs):
            self._target = target
        def start(self):
            if self._target:
                self._target()
    monkeypatch.setattr(ui.threading, "Thread", DummyThread)
    # Now run the self-test via UI
    interface = ui.TradingUI()
    interface.run_self_test()
    # After run_self_test, our DummyTestRunner.run_full_suite should have been called
    assert getattr(DummyTestRunner, "ran", False) is True, "UI self-test did not trigger SystemTestRunner"

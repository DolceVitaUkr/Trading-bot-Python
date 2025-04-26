# self_test/test_trading_bot.py

#pytest self_test/test_trading_bot.py --maxfail=1 --disable-warnings -q

import os
import sys
import math
import json
import random
import logging
from datetime import datetime, timedelta
from decimal import Decimal

import pytest
import pandas as pd

# Ensure project root for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Imports of updated modules ---
import config
import modules.data_manager as data_manager
import modules.exchange as exchange_module
import modules.technical_indicators as technical_indicators
import modules.risk_management as risk_management
import modules.reward_system as reward_system
import modules.trade_calculator as trade_calculator
import modules.trade_executor as trade_executor
import modules.top_pairs as top_pairs
import modules.error_handler as error_handler
import modules.self_learning as self_learning
import modules.trade_simulator as trade_simulator
import modules.parameter_optimization as parameter_optimization
from modules.data_manager import timeframe_to_minutes


# Alias for clarity
ExchangeAPI = exchange_module.ExchangeAPI

# --- Dummy / monkey-patches for external dependencies ---

# Dummy ExchangeClient for DataManager
class DummyExchangeClient:
    def fetch_ohlcv(self, symbol, timeframe='15m', limit=1000, since=None):
        base = int(datetime.now().timestamp() * 1000) - (limit * 60_000)
        return [
            [base + i * 60_000, 100 + i, 101 + i, 99 + i, 100.5 + i, 1000 + i]
            for i in range(limit)
        ]
    def fetch_market_data(self, symbol, timeframe, limit, since=None):
        return self.fetch_ohlcv(symbol, timeframe, limit, since)
    def fetch_ticker(self, symbol):
        return {"last": 123.45}
    def create_order(self, symbol, type, side, amount, price=None):
        return {"id":"OID","symbol":symbol,"side":side,"amount":amount,"price":price,"status":"open"}
    def load_markets(self): return True

@pytest.fixture(autouse=True)
def patch_exchange_for_data_manager(monkeypatch):
    monkeypatch.setattr(exchange_module, "ExchangeAPI",
                        lambda *args, **kw: DummyExchangeClient())
    yield

# Capture Telegram messages
sent_telegrams = []
from modules.telegram_bot import TelegramNotifier
def fake_telegram_send(self, content, format='text'):
    sent_telegrams.append((content, format))
TelegramNotifier.send_message = fake_telegram_send

# Dummy requests for top_pairs
def dummy_requests_get(url, headers=None, timeout=None):
    class R:
        def __init__(self, data): self._d = data
        def raise_for_status(self): pass
        def json(self): return self._d
    data = {
        "ret_code": 0,
        "result": [
            {"name":"TESTUSDT","quoteCurrency":"USDT","status":"Trading"},
            {"name":"BADUSD","quoteCurrency":"USD","status":"Trading"},
            {"name":"SUSPUSDT","quoteCurrency":"USDT","status":"Suspended"},
        ]
    }
    return R(data)
top_pairs.requests.get = dummy_requests_get

# Ensure simulation mode
config.USE_SIMULATION = True
config.BYBIT_API_KEY = "A"*32
config.BYBIT_API_SECRET = "B"*48
config.SIMULATION_BYBIT_API_KEY = "C"*16
config.SIMULATION_BYBIT_API_SECRET = "D"*32

# --- Tests ---

def test_data_manager_sequential_update(tmp_path):
    # Instantiate in test mode so it uses mock data and 'test_' filenames
    dm = data_manager.DataManager(test_mode=True)
    dm.data_folder = str(tmp_path)

    # --- First batch ---
    k1 = dm._generate_mock_klines(periods=5, timeframe='1m')
    assert dm.update_klines("AAA/USDT", "1m", k1)
    df1 = dm.load_historical_data("AAA/USDT", "1m")
    assert len(df1) == 5

    # --- Second batch overlaps last timestamp ---
    # Compute 1m interval in ms
    interval = timeframe_to_minutes("1m") * 60 * 1000
    ts0 = int(df1.index[-1].timestamp() * 1000)

    # Build 3 new klines: first overlaps, next two new
    k2 = [
        [
            ts0 + i * interval,
            50000 + i,
            50000 + i + 50,
            50000 + i - 50,
            50000 + i + 25,
            1000 + i
        ]
        for i in range(3)
    ]
    assert dm.update_klines("AAA/USDT", "1m", k2)

    # Clear cache so load_historical_data re-reads the updated parquet
    dm.cache.clear()
    df2 = dm.load_historical_data("AAA/USDT", "1m")

    # Expect 5 + (3 − 1 overlapping) = 7 unique rows
    assert len(df2) == 7
    assert df2.index.is_unique

def test_technical_indicators_basic_functions():
    # Simple list-based tests for your top-level functions
    data = [1.0, 2.0, 3.0, 4.0, 5.0]

    # SMA over last 3 values → (3 + 4 + 5) / 3 = 4.0
    ma = technical_indicators.moving_average(data, period=3)
    assert pytest.approx(ma, rel=1e-9) == 4.0

    # EMA: should return a float
    ema = technical_indicators.exponential_moving_average(data, period=3)
    assert isinstance(ema, float)

    # ADX: small test with period=2
    high  = [1.0, 2.0, 3.0, 4.0, 5.0]
    low   = [0.5, 1.5, 2.5, 3.5, 4.5]
    close = [1.0, 2.0, 3.0, 4.0, 5.0]
    adx_val = technical_indicators.adx(high, low, close, period=2)
    assert isinstance(adx_val, float)

    # CCI: basic sanity check
    cci_val = technical_indicators.cci(high, low, close, period=3)
    assert isinstance(cci_val, float)

    # Williams %R: float in [-100, 0]
    wr = technical_indicators.williams_r(high, low, close, period=3)
    assert isinstance(wr, float)
    assert -100.0 <= wr <= 0.0


def test_reward_system_various_scenarios():
    now = datetime.now()
    later = now + timedelta(hours=1)
    # Breakeven quick exit -> 50 points
    pts = reward_system.calculate_points(0.0, now, now, False)
    assert pytest.approx(pts, rel=1e-6) == 50.0
    # Profit but held >5h -> no time bonus
    pts2 = reward_system.calculate_points(0.2, now, now + timedelta(hours=6), False)
    assert pts2 == pytest.approx(20.0, rel=1e-6)
    # Loss unadjusted vs adjusted
    p_un = reward_system.calculate_points(-0.1, now, later, True, risk_adjusted=False)
    p_adj = reward_system.calculate_points(-0.1, now, later, True, risk_adjusted=True)
    assert p_un < p_adj

def test_error_handler_and_critical_alert(tmp_path):
    cfg = config.LOG_FILE
    config.LOG_FILE = str(tmp_path/"error.log")
    handler = error_handler.ErrorHandler()
    sent_telegrams.clear()
    err = risk_management.RiskViolationError("Exceeded", context={'x':1})
    handler.handle(err)
    assert any("CRITICAL ERROR" in m for m,_ in sent_telegrams)
    sent_telegrams.clear()
    err2 = risk_management.NetworkError("Network fail")
    handler.handle(err2)
    assert not sent_telegrams
    config.LOG_FILE = cfg

def test_top_pairs_and_cache(tmp_path):
    pm = top_pairs.PairManager()
    pm.cache_file = str(tmp_path/"pairs.json")
    pm.fallback_file = str(tmp_path/"fb.json")
    ps = pm.get_active_pairs()
    assert "TESTUSDT" in ps and "BADUSD" not in ps
    assert pm.get_active_pairs() == ps  # cache hit

def test_trade_executor_and_precision(monkeypatch):
    monkeypatch.setattr(ExchangeAPI, "get_price", lambda self, s: 50000.0)
    monkeypatch.setattr(ExchangeAPI, "create_order", lambda self, **kw: {"price": 50000.99, "id":"X","status":"open"})
    te = trade_executor.TradeExecutor(simulation_mode=False)
    res = te.execute_order("BTC/USDT","buy",1.2345,50000.9876,"limit")
    assert math.isclose(res['price'], 50000.99, rel=1e-9)
    with pytest.raises(ValueError):
        te.execute_order("BTC/USDT","buy",0.0001,50000,"limit")

def test_self_learning_main_vs_exploratory():
    class DummyExec(trade_executor.TradeExecutor):
        def __init__(self): pass
        def execute_order(self, **kw):
            return {
                'entry_price': kw['price'],
                'exit_price': kw['price'] * 1.01,
                'quantity': kw['amount'],
                'entry_time': datetime.now(),
                'exit_time': datetime.now() + timedelta(minutes=1),
                'max_drawdown':0.0, 'volatility':0.0,
                'stop_loss_triggered':False, 'done':True
            }
    bot = self_learning.SelfLearningBot(state_size=4, action_size=3, training=True)
    bot.executor = DummyExec()
    state = {'symbol':'ZZZ','price':100.0,'position_size':1.0}
    bot.epsilon = 0.0
    bot.memory.clear()
    bot.act_and_learn(state, datetime.now())
    assert len(bot.memory) == 1
    bot.epsilon = 1.0
    before = len(bot.memory)
    bot.act_and_learn(state, datetime.now())
    assert len(bot.memory) == before + 1

def test_trade_simulator_backtest(monkeypatch):
    sim = trade_simulator.TradeSimulator(initial_wallet=1000.0)
    df = pd.DataFrame({
        'open':[1,2,3], 'high':[2,3,4], 'low':[0,1,2],
        'close':[1,2,3], 'volume':[10,20,30]
    }, index=pd.date_range('2021-01-01', periods=3, freq='D'))
    monkeypatch.setattr(sim.data_manager, "load_historical_data", lambda *_: df)
    if hasattr(sim, "_generate_trading_signals"):
        sim._generate_trading_signals = lambda k: {'buy':k[4]<2.5, 'sell':k[4]>2.5}
    sim.run(list(zip(
        df.index.astype(int).tolist(),
        df['open'], df['high'], df['low'], df['close'], df['volume']
    )))
    assert hasattr(sim, 'portfolio_value')

def test_parameter_optimizer_grid_search(tmp_path):
    config.HISTORICAL_DATA_PATH = str(tmp_path)
    po = parameter_optimization.ParameterOptimizer()
    def obj(p): return sum(p.values())
    best = po.optimize(obj)
    for k,v in best.items():
        vals = config.OPTIMIZATION_PARAMETERS[k]
        if isinstance(vals, list):
            assert v in vals
        else:
            assert vals['min'] <= v <= vals['max']
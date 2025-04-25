# self_test/full_system_test.py
import logging
import os
import time
import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime, timedelta
import sys
import asyncio

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from modules.exchange import ExchangeAPI
from modules.data_manager import DataManager
from modules.trade_executor import TradeExecutor
from modules.trade_simulator import TradeSimulator
from modules.parameter_optimization import ParameterOptimizer
from modules.self_learning import SelfLearningBot
from modules.technical_indicators import TechnicalIndicators
from modules.error_handler import ErrorHandler
from modules.telegram_bot import TelegramNotifier as TelegramBot
from config import (
    USE_SIMULATION,
    SIMULATION_START_BALANCE,
    DEFAULT_TRADE_AMOUNT,
    HISTORICAL_DATA_PATH
)

class SystemTestRunner:
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
        self.test_data = {
            'symbol': 'BTC/USDT',
            'timeframe': '15m',
            'test_amount': 0.001,
            'test_price': 50000
        }
        
        self.exchange = ExchangeAPI()
        self.data_manager = DataManager()
        self.error_handler = ErrorHandler()
        self.telegram_bot = TelegramBot(disable_async=True)
        self.trade_executor = TradeExecutor(simulation_mode=USE_SIMULATION)
        self.simulator = TradeSimulator(initial_wallet=SIMULATION_START_BALANCE)
        self.optimizer = ParameterOptimizer()
        self.ai_model = SelfLearningBot()
        self.error_handler.debug_mode = True  # Enable verbose errors

    def _timed_test(func):
        def wrapper(self, *args, **kwargs):
            start = time.time() 
            result = func(self, *args, **kwargs)
            duration = time.time() - start
            self.test_results[func.__name__] = {
                'status': 'PASSED' if result else 'FAILED',
                'duration': f"{duration:.2f}s"
            }
            return result
        return wrapper

    def _cleanup_test_data(self):
        test_files = [
            f"{HISTORICAL_DATA_PATH}/btcusdt_15m.parquet",
            "optimization_checkpoint.pkl"
        ]
        for f in test_files:
            try: 
                if os.path.exists(f):
                    os.remove(f)
            except Exception as e:
                logging.warning(f"Cleanup error: {str(e)}")

    @_timed_test
    def test_historical_data_pipeline(self):
        try:
            test_symbol = "BTC/USDT"
            clean_symbol = test_symbol.replace('/', '').lower()
            
            # Generate continuous data
            interval = 15 * 60 * 1000
            start_time = int(time.time() * 1000) - (500 * interval)
            initial_klines = self.data_manager._generate_continuous_klines(500, start_time, '15m')
            
            # First update
            if not self.data_manager.update_klines(test_symbol, '15m', initial_klines):
                return False

            # Second update with gap check
            last_timestamp = initial_klines[-1][0]
            new_klines = self.data_manager._generate_continuous_klines(
                100, last_timestamp + interval, '15m'
            )
            if not self.data_manager.update_klines(test_symbol, '15m', new_klines, test_mode=True):
                return False

            # Verify merged data
            full_data = self.data_manager.load_historical_data(clean_symbol)
            return len(full_data) == 600 and full_data.index.is_monotonic_increasing
                
        except Exception as e:
            self.error_handler.log_error(e)
            return False

    @_timed_test
    def test_live_data_monitoring(self):
        try:
            test_symbol = "BTC/USDT"
            clean_symbol = test_symbol.replace("/", "").lower()
            
            interval = 15 * 60 * 1000
            start_time = int(time.time() * 1000) - (10 * interval)
            initial_klines = self.data_manager._generate_continuous_klines(5, start_time, '15m')
            
            if not self.data_manager.update_klines(test_symbol, '15m', initial_klines):
                return False

            last_ts = initial_klines[-1][0]
            new_klines = self.data_manager._generate_continuous_klines(10, last_ts + interval, '15m')
            if not self.data_manager.update_klines(test_symbol, '15m', new_klines):
                return False

            updated_data = self.data_manager.load_historical_data(clean_symbol)
            return len(updated_data) == 15
        except Exception as e:
            self.error_handler.log_error(e)
            return False
        
    @_timed_test
    def test_market_data_fetching(self):
        try:
            data = self.exchange.fetch_market_data("BTC/USDT", limit=10)
            return len(data) >= 5 and all(len(k) == 6 for k in data)
        except Exception as e:
            self.error_handler.log_error(e)
            return False

    @_timed_test
    def test_order_execution(self):
        try:
            symbol = "BTC/USDT"
            clean_symbol = symbol.replace("/", "").upper()
            current_price = self.exchange.get_current_price(clean_symbol)
            min_order = max(0.002, self.exchange.get_min_order_size(clean_symbol))
            
            # Use tighter price offset and market order
            order = self.trade_executor.execute_order(
                symbol=symbol,
                side="buy",
                amount=min_order,
                price=round(current_price * 0.9999, 2),
                order_type="market" if USE_SIMULATION else "limit"
            )
            return order['status'] in ['filled', 'pending', 'simulated']
        except Exception as e:
            self.error_handler.log_error(e)
            return False

    @_timed_test
    def test_simulation_workflow(self):
        try:
            self.simulator.trading_pairs = ["BTC/USDT"]
            initial_balance = self.simulator.wallet_balance
            
            # More volatile test data
            test_data = [
                [
                    int(time.time() * 1000) - (100 - i) * 900000,  # Timestamp
                    50000 + (i % 10) * 1000 - 500,  # Open
                    50000 + (i % 10) * 1000 + 500,  # High
                    50000 + (i % 10) * 1000 - 750,  # Low
                    50000 + (i % 10) * 1000,        # Close
                    10000 + i * 100                 # Volume
                ]
                for i in range(100)
            ]
            
            self.simulator.run(market_data=test_data)
            return abs(self.simulator.wallet_balance - initial_balance) > 10.0
        except Exception as e:
            self.error_handler.log_error(e)
            return False

    @_timed_test
    def test_parameter_optimization(self):
        try:
            original_params = self.optimizer.params
            self.optimizer.params = {
                'ma_fast': {'min': 10, 'max': 20},
                'ma_slow': {'min': 20, 'max': 30}
            }
            
            def objective(params):
                score = 100 - abs(params['ma_fast'] - 12)*10 - abs(params['ma_slow'] - 26)*5
                return score
                
            best = self.optimizer.optimize(objective)
            valid = (10 <= best['ma_fast'] <= 20) and (20 <= best['ma_slow'] <= 30)
            
            self.optimizer.params = original_params
            return valid
        except Exception as e:
            self.error_handler.log_error(e)
            return False

    @_timed_test
    def test_technical_analysis(self):
        try:
            base_prices = np.cumsum(np.random.normal(0, 50, 300)) + 50000
            test_data = pd.DataFrame({
                'open': base_prices - 25,
                'high': base_prices + 25,
                'low': base_prices - 50,
                'close': base_prices,
                'volume': np.random.randint(1000, 5000, 300)
            }, index=pd.date_range(end=datetime.now(), periods=300, freq='15min'))
            
            test_data['sma_20'] = TechnicalIndicators.moving_average(test_data['close'], 20)
            test_data['rsi_14'] = TechnicalIndicators.rsi(test_data['close'], 14)
            
            valid_sma = test_data['sma_20'].iloc[19:].notnull().all()
            valid_rsi = test_data['rsi_14'].between(0, 100).iloc[14:].all()
            
            return valid_sma and valid_rsi
        except Exception as e:
            self.error_handler.log_error(e)
            return False

    @_timed_test
    def test_ai_predictions(self):
        try:
            test_symbol = "BTC/USDT"
            clean_symbol = test_symbol.replace("/", "").lower()
            test_file = f"{HISTORICAL_DATA_PATH}/test_btcusdt_15m.parquet"

            # 1. Ensure test data exists
            if not os.path.exists(test_file) or os.path.getsize(test_file) == 0:
                self.data_manager.update_klines(
                    symbol=test_symbol,
                    timeframe='15m',
                    klines=self.data_manager._generate_continuous_klines(1000),
                    test_mode=True
                )

            # 2. Load data with retries
            klines = None
            for attempt in range(3):
                try:
                    klines = self.data_manager.load_historical_data(
                        symbol=clean_symbol,
                        timeframe='15m',
                        test_mode=True
                    )
                    if len(klines) >= 900:
                        break
                    time.sleep(0.5 * (attempt + 1))
                except FileNotFoundError:
                    if attempt == 2:
                        raise
                    self.test_historical_data_pipeline()
            else:
                raise ValueError("Failed to load sufficient data after 3 attempts")

            # 3. Prepare AI input
            closes = klines['close']
            highs = klines['high']
            lows = klines['low']
            volumes = klines['volume']

            state = {
                'price': closes.iloc[-1],
                'open': klines['open'].iloc[-1],
                'high': highs.iloc[-1],
                'low': lows.iloc[-1],
                **TechnicalIndicators.calculate_all(closes, highs, lows, volumes)
            }

            # 4. Make prediction
            test_bot = SelfLearningBot()
            prediction = test_bot.predict(state)
            
            # 5. Validate result
            valid_actions = ["buy", "sell", "hold"]
            if prediction not in valid_actions:
                raise ValueError(f"Invalid prediction: {prediction}")
            
            logging.debug(f"AI Prediction: {prediction}")
            return True

        except Exception as e:
            self.error_handler.log_error(e, {
                'test_stage': 'ai_prediction',
                'data_length': len(klines) if not klines.empty else 0
            })
            return False

    @_timed_test
    def test_error_handling(self):
        try:
            raise ValueError("Test error")
        except Exception as e:
            self.error_handler.log_error(e)
            return "Test error" in str(self.error_handler.error_log)

    @_timed_test
    def test_notification_system(self):
        try:
            self.telegram_bot.send_message("System test notification") 
            return True
        except Exception as e:
            self.error_handler.log_error(e)
            return False

    def run_full_suite(self):
        print("🚀 Starting Comprehensive System Test\n")
        self._cleanup_test_data()
        
        tests = [
            self._wrap_test(self.test_historical_data_pipeline, "Historical Data Pipeline"),
            self._wrap_test(self.test_live_data_monitoring, "Live Data Monitoring"),
            self._wrap_test(self.test_market_data_fetching, "Market Data Fetching"),
            self._wrap_test(self.test_order_execution, "Order Execution"),
            self._wrap_test(self.test_simulation_workflow, "Simulation Workflow"),
            self._wrap_test(self.test_parameter_optimization, "Parameter Optimization"),
            self._wrap_test(self.test_technical_analysis, "Technical Analysis"),
            self._wrap_test(self.test_ai_predictions, "AI Predictions"),
            self._wrap_test(self.test_error_handling, "Error Handling"),
            self._wrap_test(self.test_notification_system, "Notification System")
        ]
        
        for test in tests:
            test()
             
        print("\n📊 Test Results:")
        passed = sum(1 for r in self.test_results.values() if r['status'] == 'PASSED')
        for name, result in self.test_results.items():
            status = "✅ PASSED" if result['status'] == 'PASSED' else "❌ FAILED"
            print(f"{name.replace('_', ' ').title():<30} {status} ({result['duration']})")
        
        total_time = time.time() - self.start_time
        print(f"\n🏁 Test Complete: {passed}/{len(tests)} Passed | ⏱ Total Time: {total_time:.2f}s")
        
        self.telegram_bot.send_message("📊 Test results sent via Telegram")

    def _wrap_test(self, test_func, test_name):
        def wrapper():
            print(f"\n=== RUNNING TEST: {test_name} ===")
            try:
                result = test_func()
                status = "PASSED" if result else "FAILED"
                color = "\033[92m" if result else "\033[91m"
                print(f"{color}Test {status}\033[0m")
            except Exception as e:
                self.error_handler.log_error(e, {"test_name": test_name})
                print("\033[91mTest FAILED\033[0m")
            print("=" * 50)
        return wrapper

if __name__ == "__main__":
    runner = SystemTestRunner()
    runner.run_full_suite()
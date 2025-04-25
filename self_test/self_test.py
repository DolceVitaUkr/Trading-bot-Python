# self_test/self_test.py
import logging
import os
import time
import numpy as np
import pandas as pd

# Updated imports
from modules.exchange import ExchangeAPI
from modules.telegram_bot import TelegramNotifier
from modules.ui import TradingUI
from modules.data_manager import DataManager
from modules.error_handler import ErrorHandler
from modules.parameter_optimization import ParameterOptimizer
from modules.trade_simulator import TradeSimulator  # Updated import
from modules.self_learning import SelfLearningBot
from modules.technical_indicators import TechnicalIndicators

def test_exchange_api():
    print("Testing ExchangeAPI...")
    api = ExchangeAPI()
    data = api.fetch_market_data('BTC/USDT', '15m', limit=10)
    print("ExchangeAPI test passed" if data else "ExchangeAPI test failed")

def test_telegram_bot():
    print("Testing TelegramBot...")
    bot = TelegramNotifier()
    try:
        bot.send_message("Test message from self_test.")
        print("TelegramBot test passed")
    except Exception as e:
        print(f"TelegramBot test failed: {e}")

def test_ui():
    print("Testing TradingUI...")
    try:
        TradingUI()  # Test initialization only
        print("TradingUI test passed")
    except Exception as e:
        print(f"TradingUI test failed: {e}")

def test_data_manager():
    print("Testing DataManager...")
    try:
        dm = DataManager()
        test_symbol = "TEST_BTC_USDT"
        dummy_data = [
            [1617184800000, 100, 105, 95, 102, 1000],
            [1617188400000, 102, 108, 101, 107, 1200]
        ]
        dm.update_klines(test_symbol, '15m', dummy_data)
        df = dm.load_historical_data(test_symbol)
        print("DataManager test passed" if not df.empty else "DataManager test failed")
    except Exception as e:
        print(f"DataManager test failed: {e}")

def test_error_handler():
    print("Testing ErrorHandler...")
    try:
        raise ValueError("Test error")
    except Exception as e:
        handler = ErrorHandler()
        handler.log_error(e)
        print("ErrorHandler test passed")

def test_parameter_optimizer():
    print("Testing ParameterOptimizer...")
    try:
        optimizer = ParameterOptimizer()
        def dummy_obj(params): return sum(params.values())
        optimizer.optimize(dummy_obj)
        print("ParameterOptimizer test passed")
    except Exception as e:
        print(f"ParameterOptimizer test failed: {e}")

def test_simulation():
    print("Testing TradeSimulator...")
    try:
        sim = TradeSimulator()  # Updated class name
        result = sim.execute_order("BTC/USDT", "buy", 0.1, 50000)
        print("Simulation test passed" if result['status'] == 'simulated' else "Simulation test failed")
    except Exception as e:
        print(f"Simulation test failed: {e}")

def test_self_learning():
    print("Testing SelfLearningBot...")
    try:
        bot = SelfLearningBot()
        X = np.random.rand(10, 5)
        y = np.random.randint(0, 2, 10)
        bot.train(X, y)
        print("SelfLearning test passed")
    except Exception as e:
        print(f"SelfLearning test failed: {e}")

def test_technical_indicators():
    print("Testing TechnicalIndicators...")
    try:
        data = pd.DataFrame({'close': [100, 102, 101, 105, 107]})
        TechnicalIndicators.moving_average(data, 2)
        print("TechnicalIndicators test passed")
    except Exception as e:
        print(f"TechnicalIndicators test failed: {e}")

def main():
    logging.basicConfig(level=logging.INFO)
    print("\n=== Starting Module Tests ===")
    
    module_tests = [
        ("Exchange API", test_exchange_api),
        ("Telegram Bot", test_telegram_bot),
        ("UI Framework", test_ui),
        ("Data Manager", test_data_manager),
        ("Error Handler", test_error_handler),
        ("Parameter Optimizer", test_parameter_optimizer),
        ("Trade Simulator", test_simulation),  # Updated name
        ("Self Learning", test_self_learning),
        ("Technical Indicators", test_technical_indicators)
    ]

    for name, test in module_tests:
        print(f"\nTesting {name}...")
        try:
            test()
            time.sleep(0.5)
        except Exception as e:
            print(f"{name} test failed: {str(e)}")

    print("\n=== Module Tests Complete ===")

if __name__ == "__main__":
    main()
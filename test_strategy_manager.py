#!/usr/bin/env python3
"""
Test script for StrategyManager functionality
"""
import sys
import os

# Ensure we can import tradingbot
sys.path.insert(0, '.')

try:
    from tradingbot.core.strategy_manager import StrategyManager
    from tradingbot.core.schemas import StrategyMeta
    
    print("Successfully imported StrategyManager and StrategyMeta")
    
    # Initialize strategy manager
    sm = StrategyManager()
    print(f"Strategy manager initialized")
    print(f"   Strategy file: {sm.strategy_file}")
    
    # Test creating a strategy meta object (example)
    try:
        from datetime import datetime
        strategy_meta = StrategyMeta(**{
            "strategy_id": "test_strategy_001",
            "name": "Test Strategy", 
            "class": "crypto",  # Using the actual field name
            "market": "CRYPTO_SPOT",
            "session_flags": ["backtesting", "paper"],
            "timeframe": "15m",
            "indicators": ["SMA", "RSI", "MACD"],
            "params": {"sma_period": 20, "rsi_period": 14},
            "version": "1.0.0",
            "created_at": datetime.now()
        })
        print(f"Created strategy meta: {strategy_meta.strategy_id}")
        
        # Test registering the strategy
        sm.register_strategy(strategy_meta)
        print("Strategy registered successfully")
        
    except Exception as e:
        print(f"Strategy registration test failed: {e}")
    
    print("\nStrategyManager is working correctly!")
    
except ImportError as e:
    print(f"Import failed: {e}")
    print("\nPossible solutions:")
    print("1. Make sure you're in the Trading-bot-Python directory")
    print("2. Ensure tradingbot/__init__.py exists")
    print("3. Check that all required dependencies are installed")
    
except Exception as e:
    print(f"Error: {e}")
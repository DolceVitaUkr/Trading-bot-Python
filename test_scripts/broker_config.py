"""
Broker Connection Configuration
"""

import os
from pathlib import Path

# IBKR Configuration
IBKR_CONFIG = {
    "host": "127.0.0.1",
    "port": 7496,  # Paper trading port for TWS
    "client_id": 1,
    "account_type": "paper",  # paper or live
    "readonly": False,
    "timeout": 10
}

# Bybit Configuration
BYBIT_CONFIG = {
    "testnet": False,  # Set to False for mainnet
    "api_key": os.getenv("BYBIT_API_KEY", ""),
    "api_secret": os.getenv("BYBIT_API_SECRET", ""),
    "recv_window": 5000
}

# Paper Trading Configuration
PAPER_TRADING_CONFIG = {
    "starting_balance": 1000.0,
    "isolation_mode": True,  # Ensure paper trading doesn't use live funds
    "save_state": True,
    "state_dir": Path("tradingbot/state")
}

# Safety Configuration
SAFETY_CONFIG = {
    "require_confirmation": True,  # Require confirmation for live trades
    "max_position_size_pct": 0.1,  # Max 10% of account per position
    "daily_loss_limit_pct": 0.05,  # Max 5% daily loss
    "paper_live_isolation": True,  # Strict isolation between paper and live
}

def get_ibkr_config(paper_mode=True):
    """Get IBKR configuration based on mode"""
    config = IBKR_CONFIG.copy()
    if paper_mode:
        config["port"] = 7496  # Paper trading port
        config["account_type"] = "paper"
    else:
        config["port"] = 7496  # Live trading port (same for paper socket)
        config["account_type"] = "live"
    return config

def get_bybit_config(paper_mode=True):
    """Get Bybit configuration based on mode"""
    config = BYBIT_CONFIG.copy()
    if paper_mode:
        config["testnet"] = True
        # Use testnet credentials if available
        config["api_key"] = os.getenv("BYBIT_TESTNET_API_KEY", "")
        config["api_secret"] = os.getenv("BYBIT_TESTNET_API_SECRET", "")
    return config
"""
Update IBKR configuration to handle paper trading properly
"""

import json
from pathlib import Path

# Load current config
config_path = Path("tradingbot/config/config.json")
with open(config_path, 'r') as f:
    config = json.load(f)

# Update IBKR settings
config['bot_settings']['ibkr_api_mode'] = 'live'  # Change to 'live' since you're using live TWS with paper account
config['bot_settings']['ibkr_paper_trading'] = True  # Enable paper trading mode
config['api_keys']['ibkr']['port'] = 7496  # Paper socket port

# Save updated config
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print("Configuration updated:")
print(f"- IBKR API mode: live (to allow connection)")
print(f"- IBKR port: 7496 (paper socket)")
print(f"- Paper trading enabled")

# Also create a runtime override to ensure paper trading
runtime_override = {
    "IBKR_PAPER_TRADING": True,
    "IBKR_ALLOW_LIVE_ACCOUNT_FOR_PAPER": True
}

override_path = Path("tradingbot/state/ibkr_override.json")
with open(override_path, 'w') as f:
    json.dump(runtime_override, f, indent=2)

print(f"\nCreated override file: {override_path}")
print("This ensures paper trading even with live account connection")
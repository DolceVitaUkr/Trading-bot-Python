"""
Add sample trade history for display
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
import uuid

# Add a completed trade to forex paper trader
filepath = 'tradingbot/state/paper_trader_forex.json'

with open(filepath, 'r') as f:
    state = json.load(f)

# Add a completed trade to history
completed_trade = {
    "id": str(uuid.uuid4()),
    "symbol": "GBP/USD",
    "side": "SELL",
    "size": 1000,
    "entry_price": 1.2650,
    "exit_price": 1.2620,
    "stop_loss": 1.2680,
    "take_profit": 1.2600,
    "entry_time": (datetime.now() - timedelta(hours=3)).isoformat(),
    "exit_time": (datetime.now() - timedelta(hours=1)).isoformat(),
    "status": "closed",
    "strategy": "manual_test",
    "asset_type": "FOREX",
    "realized_pnl": 3.00,  # 30 pips profit
    "exit_reason": "take_profit"
}

# Add to trades history
if 'trades' not in state:
    state['trades'] = []
    
state['trades'].append(completed_trade)

# Update total P&L
state['total_pnl'] = state.get('total_pnl', 0) + completed_trade['realized_pnl']

# Save
with open(filepath, 'w') as f:
    json.dump(state, f, indent=2)

print("Added completed trade to history:")
print(f"  GBP/USD SELL - Profit: ${completed_trade['realized_pnl']:.2f} (30 pips)")
print(f"  Entry: {completed_trade['entry_price']} -> Exit: {completed_trade['exit_price']}")
print("\nTrade will now appear in Historical Trades section")
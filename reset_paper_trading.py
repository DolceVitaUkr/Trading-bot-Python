"""Reset all paper trading data to start fresh"""

import json
from pathlib import Path
from datetime import datetime

def reset_paper_trader(asset_type):
    """Reset paper trader to initial state"""
    state_file = Path(f"tradingbot/state/paper_trader_{asset_type}.json")
    
    initial_state = {
        "balance": 1000.0,
        "starting_balance": 1000.0,
        "positions": [],
        "trades": [],
        "pnl_history": [
            {
                "timestamp": datetime.now().isoformat(),
                "balance": 1000.0
            }
        ],
        "violations_log": []
    }
    
    if state_file.exists():
        print(f"Resetting {asset_type} paper trader...")
        with open(state_file, 'w') as f:
            json.dump(initial_state, f, indent=2)
        print(f"[OK] {asset_type} paper trader reset to $1000.00")
    else:
        print(f"Creating new {asset_type} paper trader state...")
        state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(state_file, 'w') as f:
            json.dump(initial_state, f, indent=2)
        print(f"[OK] {asset_type} paper trader created with $1000.00")

def main():
    print("=== Resetting All Paper Trading Data ===\n")
    
    assets = ['crypto', 'futures', 'forex', 'forex_options']
    
    for asset in assets:
        reset_paper_trader(asset)
    
    # Also reset runtime state
    runtime_file = Path("tradingbot/state/runtime.json")
    if runtime_file.exists():
        print("\nResetting runtime state...")
        runtime_state = {
            "trading": {},
            "positions": {},
            "last_update": datetime.now().isoformat()
        }
        with open(runtime_file, 'w') as f:
            json.dump(runtime_state, f, indent=2)
        print("[OK] Runtime state reset")
    
    print("\n=== All Paper Trading Data Reset Successfully ===")
    print("All paper traders now have:")
    print("- Starting balance: $1000.00")
    print("- No open positions")
    print("- No trade history")
    print("- Fresh PnL history")

if __name__ == "__main__":
    main()
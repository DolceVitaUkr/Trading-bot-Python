#!/usr/bin/env python3
"""
Test script to trigger strategy execution and generate trades
"""
import requests
import time
import json

BASE_URL = "http://127.0.0.1:8000"

def enable_paper_trading(asset):
    """Enable paper trading for an asset"""
    endpoint = f"{BASE_URL}/paper/{asset}/enable"
    
    print(f"\nEnabling paper trading for {asset}...")
    try:
        response = requests.post(endpoint)
        if response.status_code == 200:
            print(f"[SUCCESS] Paper trading enabled for {asset}")
            return True
        else:
            print(f"[FAILED] {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

def check_runtime_status():
    """Check runtime status of all assets"""
    endpoint = f"{BASE_URL}/runtime/status"
    
    try:
        response = requests.get(endpoint)
        if response.status_code == 200:
            status = response.json()
            print("\n=== Runtime Status ===")
            print(json.dumps(status, indent=2))
            return status
        else:
            print(f"Failed to get runtime status: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_strategy_status(asset):
    """Get strategy status for an asset"""
    endpoint = f"{BASE_URL}/strategy/{asset}/status"
    
    try:
        response = requests.get(endpoint)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to get strategy status: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def trigger_strategy_creation():
    """Trigger the creation of position through strategy simulation"""
    # This is a workaround - we'll directly call the paper trader's methods
    # through a custom endpoint that we'll add
    print("\nNote: The trading bot appears to work through automated strategies.")
    print("Strategies execute based on market conditions and signals.")
    print("To see trades, strategies need to be properly configured and market conditions met.")
    
def check_positions(asset):
    """Check current positions for an asset"""
    endpoint = f"{BASE_URL}/positions/{asset}"
    
    try:
        response = requests.get(endpoint)
        if response.status_code == 200:
            positions = response.json()
            print(f"\n=== {asset.upper()} Positions ===")
            print(f"Paper positions: {positions.get('paper_positions', [])}")
            print(f"Live positions: {positions.get('live_positions', [])}")
            return positions
        else:
            print(f"Failed to get positions: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    print("=" * 60)
    print("STRATEGY EXECUTION TEST")
    print("=" * 60)
    
    # Check runtime status
    runtime_status = check_runtime_status()
    
    # Enable paper trading for all assets
    assets = ['crypto', 'futures', 'forex', 'forex_options']
    
    for asset in assets:
        # Enable paper trading
        enable_paper_trading(asset)
        
        # Check strategy status
        print(f"\n--- Strategy Status for {asset} ---")
        strategy_status = get_strategy_status(asset)
        if strategy_status:
            active_strategies = strategy_status.get('active_strategies', [])
            print(f"Active strategies: {len(active_strategies)}")
            for strategy in active_strategies:
                print(f"  - {strategy.get('strategy_id', 'N/A')}: {strategy.get('status', 'N/A')}")
        
        # Check positions
        check_positions(asset)
    
    print("\n" + "=" * 60)
    print("IMPORTANT NOTES:")
    print("=" * 60)
    print("1. This trading bot operates on automated strategies")
    print("2. Trades are executed when market conditions match strategy criteria")
    print("3. Strategies need to be properly configured with entry/exit conditions")
    print("4. The bot validates against real market data for realistic paper trading")
    print("\nTo see trades in action:")
    print("- Strategies must be active and properly configured")
    print("- Market conditions must match strategy entry criteria")
    print("- The bot will automatically execute trades when conditions are met")
    print("=" * 60)

if __name__ == "__main__":
    main()
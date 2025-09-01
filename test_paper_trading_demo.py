#!/usr/bin/env python3
"""
Test script to demonstrate paper trading functionality
"""
import requests
import time
import json

BASE_URL = "http://127.0.0.1:8000"

def execute_paper_trade(asset, symbol, side, amount, price=None):
    """Execute a paper trade"""
    endpoint = f"{BASE_URL}/trade/{asset}/paper"
    
    payload = {
        "symbol": symbol,
        "side": side,
        "amount": amount,
        "order_type": "market" if price is None else "limit",
        "price": price
    }
    
    print(f"\nExecuting {side} order for {symbol} on {asset}...")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(endpoint, json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"[SUCCESS] Order executed successfully!")
            print(f"Result: {json.dumps(result, indent=2)}")
            return True
        else:
            print(f"[FAILED] Order failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"[ERROR] Request failed: {e}")
        return False

def get_asset_status(asset):
    """Get current status of an asset"""
    endpoint = f"{BASE_URL}/asset/{asset}/status"
    
    try:
        response = requests.get(endpoint)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to get status for {asset}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Request failed: {e}")
        return None

def main():
    print("=" * 60)
    print("PAPER TRADING DEMO")
    print("=" * 60)
    
    # Test trades for different assets
    test_trades = [
        # Crypto spot trades
        {"asset": "crypto", "symbol": "BTCUSDT", "side": "buy", "amount": 0.01},
        {"asset": "crypto", "symbol": "ETHUSDT", "side": "buy", "amount": 0.1},
        
        # Crypto futures trades
        {"asset": "futures", "symbol": "BTCUSDT", "side": "buy", "amount": 0.005},
        
        # Wait and then close some positions
        {"wait": 5},
        
        # Close some positions
        {"asset": "crypto", "symbol": "BTCUSDT", "side": "sell", "amount": 0.01},
        {"asset": "crypto", "symbol": "ETHUSDT", "side": "sell", "amount": 0.05},
        {"asset": "futures", "symbol": "BTCUSDT", "side": "sell", "amount": 0.005},
    ]
    
    for trade in test_trades:
        if "wait" in trade:
            print(f"\nWaiting {trade['wait']} seconds...")
            time.sleep(trade['wait'])
            continue
            
        # Get current status before trade
        print(f"\n--- Status before {trade['side']} {trade['symbol']} ---")
        status = get_asset_status(trade['asset'])
        if status:
            paper_wallet = status.get('paper_wallet', {})
            print(f"Balance: ${paper_wallet.get('balance', 0):.2f}")
            print(f"P&L: ${paper_wallet.get('pnl', 0):.2f} ({paper_wallet.get('pnl_percent', 0):.2f}%)")
        
        # Execute trade
        success = execute_paper_trade(
            trade['asset'],
            trade['symbol'],
            trade['side'],
            trade['amount']
        )
        
        if success:
            # Wait a bit for the trade to process
            time.sleep(2)
            
            # Get status after trade
            print(f"\n--- Status after trade ---")
            status = get_asset_status(trade['asset'])
            if status:
                paper_wallet = status.get('paper_wallet', {})
                print(f"Balance: ${paper_wallet.get('balance', 0):.2f}")
                print(f"P&L: ${paper_wallet.get('pnl', 0):.2f} ({paper_wallet.get('pnl_percent', 0):.2f}%)")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE - Check the dashboard for updated metrics!")
    print("=" * 60)

if __name__ == "__main__":
    main()
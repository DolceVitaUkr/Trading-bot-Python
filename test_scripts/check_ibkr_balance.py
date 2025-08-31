"""
Check IBKR balance via API
"""

import requests
import json

# Get forex status which includes IBKR wallet data
print("Checking IBKR balance via API...")

try:
    response = requests.get("http://localhost:8000/asset/forex/status")
    data = response.json()
    
    print(f"\nConnection Status: {data['connection_status']}")
    
    if data['connection_status'] == 'connected':
        live_wallet = data['live_wallet']
        print(f"\nLive IBKR Wallet:")
        print(f"  Balance: ${live_wallet.get('balance', 0):.2f}")
        print(f"  Available: ${live_wallet.get('available_balance', 0):.2f}")
        print(f"  Used in Positions: ${live_wallet.get('used_in_positions', 0):.2f}")
        print(f"  P&L: ${live_wallet.get('pnl', 0):.2f}")
        
        # Also get IBKR connection details
        conn_response = requests.get("http://localhost:8000/broker/status")
        if conn_response.status_code == 200:
            conn_data = conn_response.json()
            ibkr_info = conn_data.get('ibkr', {})
            print(f"\nIBKR Connection Info:")
            print(f"  Status: {ibkr_info.get('status', 'unknown')}")
            print(f"  Account: {ibkr_info.get('account', 'N/A')}")
            print(f"  Is Paper: {ibkr_info.get('is_paper', 'N/A')}")
    else:
        print("IBKR is not connected")
        
    # Check paper wallet too
    paper_wallet = data['paper_wallet']
    print(f"\nPaper Wallet (Forex):")
    print(f"  Balance: ${paper_wallet.get('balance', 0):.2f}")
    print(f"  P&L: ${paper_wallet.get('pnl', 0):.2f} ({paper_wallet.get('pnl_percent', 0):.1f}%)")
    
except Exception as e:
    print(f"Error: {e}")
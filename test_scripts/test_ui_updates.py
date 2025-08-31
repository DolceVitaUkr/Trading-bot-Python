#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test script to verify UI updates when trading is active."""

import requests
import time
import json

BASE_URL = "http://127.0.0.1:8000"

def test_paper_trading_toggle():
    """Test starting paper trading and checking UI updates."""
    print("Testing paper trading toggle and UI updates...")
    
    # Test for crypto asset
    asset = "crypto"
    
    # 1. Check initial status
    print(f"\n1. Checking initial status for {asset}...")
    status_resp = requests.get(f"{BASE_URL}/asset/{asset}/status")
    if status_resp.ok:
        status = status_resp.json()
        print(f"   Paper trading active: {status.get('paper_trading_active', False)}")
        print(f"   Paper balance: ${status.get('paper_wallet', {}).get('balance', 0):.2f}")
    
    # 2. Start paper trading
    print(f"\n2. Starting paper trading for {asset}...")
    start_resp = requests.post(f"{BASE_URL}/asset/{asset}/start/paper")
    if start_resp.ok:
        print("   [OK] Paper trading started successfully")
    else:
        print(f"   [ERROR] Failed to start: {start_resp.text}")
        return
    
    # 3. Wait a bit for trading to start
    print("\n3. Waiting for trading activity...")
    time.sleep(5)
    
    # 4. Check for positions
    print(f"\n4. Checking for positions...")
    positions_resp = requests.get(f"{BASE_URL}/asset/{asset}/positions")
    if positions_resp.ok:
        positions_data = positions_resp.json()
        paper_positions = positions_data.get('paper', {}).get('positions', [])
        print(f"   Active positions: {len(paper_positions)}")
        if paper_positions:
            for pos in paper_positions[:3]:  # Show first 3
                print(f"   - {pos.get('symbol')}: {pos.get('side')} ${pos.get('size', 0):.2f}")
    
    # 5. Check activity log
    print(f"\n5. Checking activity log...")
    activity_resp = requests.get(f"{BASE_URL}/activity/recent")
    if activity_resp.ok:
        activities = activity_resp.json().get('activities', [])
        crypto_activities = [a for a in activities if a.get('source', '').lower() == 'crypto']
        print(f"   Recent activities: {len(crypto_activities)}")
        for activity in crypto_activities[:5]:  # Show first 5
            print(f"   - [{activity.get('type')}] {activity.get('message')}")
    
    # 6. Check updated status
    print(f"\n6. Checking updated status...")
    status_resp = requests.get(f"{BASE_URL}/asset/{asset}/status")
    if status_resp.ok:
        status = status_resp.json()
        paper_wallet = status.get('paper_wallet', {})
        print(f"   Paper trading active: {status.get('paper_trading_active', False)}")
        print(f"   Paper balance: ${paper_wallet.get('balance', 0):.2f}")
        print(f"   P&L: ${paper_wallet.get('pnl', 0):.2f} ({paper_wallet.get('pnl_percent', 0):.2f}%)")
    
    # 7. Stop paper trading
    print(f"\n7. Stopping paper trading for {asset}...")
    stop_resp = requests.post(f"{BASE_URL}/asset/{asset}/stop/paper")
    if stop_resp.ok:
        print("   [OK] Paper trading stopped successfully")
    else:
        print(f"   [ERROR] Failed to stop: {stop_resp.text}")
    
    print("\n[OK] Test completed!")

if __name__ == "__main__":
    try:
        # Check if server is running
        print("Checking if server is running...")
        ping_resp = requests.get(f"{BASE_URL}/ping")
        if ping_resp.ok:
            print("[OK] Server is running")
        else:
            print("[ERROR] Server is not responding")
            exit(1)
            
        test_paper_trading_toggle()
        
    except requests.exceptions.ConnectionError:
        print("[ERROR] Could not connect to server. Make sure it's running on http://127.0.0.1:8000")
    except Exception as e:
        print(f"[ERROR] Error: {e}")
"""Test UI functionality including toggles and data updates."""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_api_endpoints():
    """Test all API endpoints."""
    print("Testing API Endpoints")
    print("=" * 60)
    
    endpoints = [
        ("/stats/global", "Global Stats"),
        ("/activity/recent", "Recent Activity"),
        ("/brokers/status", "Brokers Status"),
        ("/trading/status", "Trading Status"),
        ("/portfolio/positions", "Portfolio Positions"),
        ("/ping", "Server Ping"),
    ]
    
    for endpoint, name in endpoints:
        try:
            response = requests.get(BASE_URL + endpoint)
            if response.status_code == 200:
                data = response.json()
                print(f"[OK] {name}: Status {response.status_code}")
                if endpoint == "/stats/global":
                    # Check data structure
                    for key in ["crypto_paper", "crypto_live", "forex_paper", "forex_live", "futures_paper", "futures_live"]:
                        if key in data:
                            section = data[key]
                            print(f"  - {key}: positions={len(section.get('positions', []))}, "
                                  f"activities={len(section.get('activities', []))}, "
                                  f"trades={len(section.get('trades', []))}")
            else:
                print(f"[FAIL] {name}: Status {response.status_code}")
        except Exception as e:
            print(f"[ERROR] {name}: {str(e)}")
    
    print()

def test_toggle_functionality():
    """Test paper/live toggle functionality."""
    print("Testing Toggle Functionality")
    print("=" * 60)
    
    # Test starting paper trading for crypto
    try:
        response = requests.post(f"{BASE_URL}/asset/crypto/start/paper")
        if response.status_code == 200:
            print(f"[OK] Start crypto paper trading: {response.json()}")
        else:
            print(f"[FAIL] Start failed: Status {response.status_code}")
    except Exception as e:
        print(f"[ERROR] Toggle test failed: {str(e)}")
    
    # Test stopping paper trading for crypto
    try:
        response = requests.post(f"{BASE_URL}/asset/crypto/stop/paper")
        if response.status_code == 200:
            print(f"[OK] Stop crypto paper trading: {response.json()}")
        else:
            print(f"[FAIL] Stop failed: Status {response.status_code}")
    except Exception as e:
        print(f"[ERROR] Toggle test failed: {str(e)}")
    
    print()

def test_emergency_stop():
    """Test emergency stop functionality."""
    print("Testing Emergency Stop")
    print("=" * 60)
    
    try:
        response = requests.post(f"{BASE_URL}/emergency/stop")
        if response.status_code == 200:
            print(f"[OK] Emergency stop: {response.json()}")
        else:
            print(f"[FAIL] Emergency stop: Status {response.status_code}")
    except Exception as e:
        print(f"[ERROR] Emergency stop test failed: {str(e)}")
    
    print()

def test_data_updates():
    """Test real-time data updates."""
    print("Testing Data Updates")
    print("=" * 60)
    
    print("Checking for data updates over 5 seconds...")
    
    # Get initial state
    response1 = requests.get(f"{BASE_URL}/stats/global")
    data1 = response1.json() if response1.status_code == 200 else {}
    
    time.sleep(5)
    
    # Get state after 5 seconds
    response2 = requests.get(f"{BASE_URL}/stats/global")
    data2 = response2.json() if response2.status_code == 200 else {}
    
    # Check for any changes
    changes_detected = False
    for key in data1:
        if key in data2:
            if data1[key] != data2[key]:
                changes_detected = True
                print(f"[OK] Data changed for {key}")
    
    if not changes_detected:
        print("[INFO] No data changes detected in 5 seconds (this is normal if no trading is active)")
    
    print()

def test_websocket_connection():
    """Test WebSocket connectivity."""
    print("Testing WebSocket Connection")
    print("=" * 60)
    
    # Check if WebSocket endpoints are defined
    try:
        response = requests.get(BASE_URL)
        if "WebSocket" in response.text or "ws://" in response.text:
            print("[OK] WebSocket references found in HTML")
        else:
            print("[INFO] No WebSocket implementation detected")
    except Exception as e:
        print(f"[ERROR] WebSocket test failed: {str(e)}")
    
    print()

def main():
    """Run all functionality tests."""
    print("\nTrading Bot UI Functionality Test")
    print("=" * 60)
    print()
    
    # Run all tests
    test_api_endpoints()
    test_toggle_functionality()
    test_emergency_stop()
    test_data_updates()
    test_websocket_connection()
    
    print("\nTest Summary")
    print("=" * 60)
    print("Please manually verify in your browser:")
    print("1. The dashboard displays in 2 columns")
    print("2. Paper/Live toggle switches work when clicked")
    print("3. Emergency Stop button shows confirmation dialog")
    print("4. Shutdown button works")
    print("5. Tables are sortable and searchable")
    print("6. Activity logs update in real-time")
    print("\nDashboard URL: http://localhost:8000")

if __name__ == "__main__":
    main()
"""Test UI data display by making requests to the server"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_endpoints():
    """Test all endpoints and display results"""
    
    endpoints = [
        "/stats/global",
        "/activity/recent", 
        "/asset/crypto/status",
        "/asset/crypto/positions",
        "/asset/crypto/strategies",
        "/asset/futures/status",
        "/asset/futures/positions",
        "/asset/forex/status",
        "/asset/forex/positions"
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(BASE_URL + endpoint)
            if response.status_code == 200:
                data = response.json()
                print(f"\n[OK] {endpoint}")
                print(f"Response: {json.dumps(data, indent=2)[:200]}...")
            else:
                print(f"\n[FAIL] {endpoint} - Status: {response.status_code}")
        except Exception as e:
            print(f"\n[ERROR] {endpoint} - Error: {e}")
    
    # Test the main page
    try:
        response = requests.get(BASE_URL)
        if response.status_code == 200:
            print(f"\n[OK] Main dashboard page loaded successfully")
            print(f"HTML length: {len(response.text)} characters")
        else:
            print(f"\n[FAIL] Main page - Status: {response.status_code}")
    except Exception as e:
        print(f"\n[ERROR] Main page - Error: {e}")

def simulate_ui_polling():
    """Simulate the UI polling for data"""
    print("\n\nSimulating UI polling pattern...")
    
    for i in range(3):
        print(f"\n--- Poll #{i+1} ---")
        
        # Global stats
        response = requests.get(BASE_URL + "/stats/global")
        if response.status_code == 200:
            data = response.json()
            print(f"Total PnL: ${data.get('total_pnl', 0):.2f}")
            print(f"Active Assets: {data.get('active_assets', 0)}")
            print(f"Total Positions: {data.get('total_positions', 0)}")
        
        # Crypto positions
        response = requests.get(BASE_URL + "/asset/crypto/positions")
        if response.status_code == 200:
            data = response.json()
            paper_positions = data.get('paper_trading', {}).get('positions', [])
            print(f"Crypto Paper Positions: {len(paper_positions)}")
            if paper_positions:
                for pos in paper_positions[:2]:  # Show first 2
                    print(f"  - {pos['symbol']}: {pos['side']} P&L: ${pos['pnl']:.2f} ({pos['pnl_pct']:.2f}%)")
        
        time.sleep(2)

if __name__ == "__main__":
    print("Testing Trading Bot UI Data Flow")
    print("=" * 50)
    
    test_endpoints()
    simulate_ui_polling()
    
    print("\n\nTest complete! Check browser console for JavaScript logs.")
    print("Open http://localhost:8000 in your browser")
    print("Press F12 to open DevTools and check Console tab")
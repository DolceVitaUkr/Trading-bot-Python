"""
Trigger IBKR connection via API endpoint
"""

import requests
import time

# Call the broker status endpoint which triggers connection
print("Triggering IBKR connection...")

try:
    # This endpoint will attempt to connect IBKR
    response = requests.post("http://localhost:8000/ibkr/connect")
    data = response.json()
    
    print(f"Response: {data}")
    
    # Wait a bit and check forex status
    time.sleep(2)
    
    # Check if IBKR is connected for forex
    forex_response = requests.get("http://localhost:8000/asset/forex/status")
    forex_data = forex_response.json()
    
    if forex_data.get("connection_status") == "connected":
        print("[SUCCESS] IBKR is now connected!")
        print(f"Live wallet balance: ${forex_data['live_wallet']['balance']:.2f}")
    else:
        print(f"[INFO] IBKR connection status: {forex_data.get('connection_status')}")
        print("Please ensure TWS is running and API is enabled on port 7496")
        
except Exception as e:
    print(f"[ERROR] Failed to trigger connection: {e}")
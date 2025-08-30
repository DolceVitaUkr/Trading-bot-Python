"""
Test IBKR connection to fix offline status
"""
import asyncio
from ib_insync import IB
import json

async def test_and_connect_ibkr():
    """Test IBKR connection and ensure it shows online."""
    ib = IB()
    
    try:
        # Read config to get IBKR settings
        with open('tradingbot/config/config.json', 'r') as f:
            config = json.load(f)
        
        ibkr_config = config.get('ibkr', {})
        host = ibkr_config.get('host', '127.0.0.1')
        port = ibkr_config.get('port', 7496)
        client_id = ibkr_config.get('client_id', 1)
        
        print(f"Connecting to IBKR at {host}:{port} with client ID {client_id}...")
        
        # Connect to IBKR
        await ib.connectAsync(host, port, clientId=client_id)
        print("[SUCCESS] Connected to IBKR")
        
        # Get account info
        account_values = ib.accountValues()
        for av in account_values:
            if av.tag == 'NetLiquidation' and av.currency in ['BASE', 'USD']:
                print(f"Account {av.account}: ${av.value}")
        
        # Keep connection alive for a bit
        await asyncio.sleep(2)
        
        # Call the API endpoint to update connection
        import requests
        response = requests.post('http://localhost:8000/ibkr/connect')
        print(f"API Response: {response.json()}")
        
        # Check status endpoint
        status_response = requests.get('http://localhost:8000/api/status')
        status_data = status_response.json()
        print(f"IBKR Status: {status_data.get('brokers', {}).get('ibkr', {})}")
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
    finally:
        if ib.isConnected():
            ib.disconnect()
            print("Disconnected from IBKR")

if __name__ == '__main__':
    asyncio.run(test_and_connect_ibkr())
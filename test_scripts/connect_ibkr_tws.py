"""
Connect IBKR TWS for the running bot
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tradingbot.brokers.connectibkrapi import IBKRConnectionManager

# Access the singleton instance used by the bot
from tradingbot.ui.app import ibkr_connection_manager


async def connect_ibkr():
    """Connect to IBKR TWS"""
    print("\n" + "="*50)
    print("Connecting to IBKR TWS for Live Wallet Display")
    print("="*50)
    
    if not ibkr_connection_manager:
        print("[ERROR] IBKR connection manager not initialized")
        return False
    
    try:
        # Connect to TWS
        print("Connecting to TWS on port 7496...")
        await ibkr_connection_manager.connect_tws()
        print("[OK] Connected to IBKR TWS")
        
        # Get account info
        accounts = ibkr_connection_manager.ib.managedAccounts()
        if accounts:
            account = accounts[0]
            is_paper = account.startswith('D')
            print(f"Account: {account} ({'PAPER' if is_paper else 'LIVE'})")
            
            # Get wallet data
            wallet_data = await ibkr_connection_manager.get_wallet_data()
            print(f"Balance: ${wallet_data['balance']:,.2f}")
            print(f"Available: ${wallet_data['available_balance']:,.2f}")
            
        print("\n[SUCCESS] IBKR TWS connected and will display in dashboard")
        print("The dashboard will now show live IBKR wallet balance for Forex/Options")
        
        # Keep connection alive
        print("\nKeeping connection alive. Press Ctrl+C to disconnect...")
        while True:
            await asyncio.sleep(10)
            if not ibkr_connection_manager.is_connected():
                print("[WARN] Connection lost, attempting to reconnect...")
                await ibkr_connection_manager.connect_tws()
            
    except KeyboardInterrupt:
        print("\n[INFO] Disconnecting from IBKR...")
        await ibkr_connection_manager.disconnect_tws()
        print("[OK] Disconnected")
    except Exception as e:
        print(f"[ERROR] Failed to connect: {e}")
        return False


if __name__ == "__main__":
    asyncio.run(connect_ibkr())
"""
Debug IBKR balance fetching
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tradingbot.brokers.connectibkrapi import IBKRConnectionManager


async def debug_ibkr_balance():
    """Debug IBKR balance fetching"""
    print("\n" + "="*50)
    print("Debugging IBKR Balance")
    print("="*50)
    
    ibkr = IBKRConnectionManager()
    
    try:
        # Use different client ID to avoid conflict
        print("Connecting with client ID 2...")
        await ibkr.ib.connectAsync(
            host="127.0.0.1",
            port=7496,
            clientId=2,
            timeout=10
        )
        print("[OK] Connected")
        
        # Get accounts
        accounts = ibkr.ib.managedAccounts()
        print(f"\nAccounts: {accounts}")
        
        if accounts:
            account = accounts[0]
            
            # Get account summary
            print(f"\nFetching account summary for {account}...")
            summary = await ibkr.ib.accountSummaryAsync(account)
            
            print("\nAccount Summary Items:")
            print("-" * 80)
            
            # Group by tag
            summary_dict = {}
            for item in summary:
                key = f"{item.tag} ({item.currency})"
                summary_dict[key] = item.value
            
            # Sort and display
            for key in sorted(summary_dict.keys()):
                print(f"{key:50} = {summary_dict[key]:>20}")
            
            # Extract key values
            print("\n" + "="*50)
            print("Key Account Values:")
            print("="*50)
            
            for item in summary:
                if item.tag in ['NetLiquidation', 'TotalCashBalance', 'AvailableFunds', 'BuyingPower']:
                    print(f"{item.tag:20} ({item.currency:4}): ${float(item.value):,.2f}")
            
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        if ibkr.ib.isConnected():
            ibkr.ib.disconnect()
            print("\n[INFO] Disconnected")


if __name__ == "__main__":
    asyncio.run(debug_ibkr_balance())
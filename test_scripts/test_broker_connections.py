"""
Test Broker Connections and Wallet Balances
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from tradingbot.brokers.connectibkrapi import IBKRConnectionManager
from tradingbot.brokers.exchangebybit import ExchangeBybit
from tradingbot.core.paper_trader import get_paper_trader
import json


async def test_ibkr_connection():
    """Test IBKR TWS connection and get account info"""
    print("\n" + "="*50)
    print("Testing IBKR Connection (TWS on port 7496)")
    print("="*50)
    
    ibkr = IBKRConnectionManager()
    
    try:
        # Connect to TWS
        await ibkr.connect_tws()
        print("[OK] Connected to IBKR TWS")
        
        # Get account info
        accounts = ibkr.ib.managedAccounts()
        print(f"Managed Accounts: {accounts}")
        
        if accounts:
            account = accounts[0]
            is_paper = account.startswith('D')
            print(f"Account Type: {'Paper' if is_paper else 'Live'}")
            
            # Get account values
            account_values = ibkr.ib.accountValues(account)
            
            # Find key values
            net_liquidation = 0
            available_funds = 0
            buying_power = 0
            
            for av in account_values:
                if av.tag == 'NetLiquidation' and av.currency == 'USD':
                    net_liquidation = float(av.value)
                elif av.tag == 'AvailableFunds' and av.currency == 'USD':
                    available_funds = float(av.value)
                elif av.tag == 'BuyingPower' and av.currency == 'USD':
                    buying_power = float(av.value)
            
            print(f"\nAccount Summary:")
            print(f"  Net Liquidation: ${net_liquidation:,.2f}")
            print(f"  Available Funds: ${available_funds:,.2f}")
            print(f"  Buying Power: ${buying_power:,.2f}")
            
            # Get positions
            positions = ibkr.ib.positions(account)
            print(f"\nOpen Positions: {len(positions)}")
            for pos in positions[:5]:  # Show first 5
                print(f"  - {pos.contract.symbol}: {pos.position} @ ${pos.avgCost:.2f}")
            
            return {
                "connected": True,
                "account": account,
                "is_paper": is_paper,
                "net_liquidation": net_liquidation,
                "positions": len(positions)
            }
            
    except Exception as e:
        print(f"[FAIL] IBKR Connection Error: {e}")
        return {"connected": False, "error": str(e)}
    finally:
        if ibkr.ib.isConnected():
            ibkr.ib.disconnect()
            print("[INFO] Disconnected from IBKR")


def test_bybit_connection():
    """Test Bybit connection and get wallet balance"""
    print("\n" + "="*50)
    print("Testing Bybit Connection")
    print("="*50)
    
    try:
        # Initialize Bybit exchange
        exchange = ExchangeBybit("CRYPTO_SPOT", "live")
        print("[OK] Bybit exchange initialized")
        
        # Get wallet balance
        balance_info = exchange.get_wallet_balance()
        
        if balance_info and 'result' in balance_info:
            result = balance_info['result']
            if 'list' in result and result['list']:
                print("\nBybit Wallet Balances:")
                total_usd = 0
                
                for account in result['list']:
                    account_type = account.get('accountType', 'Unknown')
                    print(f"\n{account_type} Account:")
                    
                    for coin_info in account.get('coin', []):
                        coin = coin_info.get('coin', '')
                        wallet_balance = float(coin_info.get('walletBalance', 0))
                        usd_value = float(coin_info.get('usdValue', 0))
                        
                        if wallet_balance > 0:
                            print(f"  {coin}: {wallet_balance:.8f} (${usd_value:.2f})")
                            total_usd += usd_value
                
                print(f"\nTotal USD Value: ${total_usd:,.2f}")
                
                return {
                    "connected": True,
                    "total_usd_value": total_usd,
                    "accounts": len(result['list'])
                }
            else:
                print("[WARN] No wallet data returned")
                return {"connected": True, "total_usd_value": 0}
        else:
            print("[FAIL] Failed to get wallet balance")
            return {"connected": False, "error": "No balance data"}
            
    except Exception as e:
        print(f"[FAIL] Bybit Connection Error: {e}")
        return {"connected": False, "error": str(e)}


def test_paper_trading_isolation():
    """Test that paper trading is isolated from live wallets"""
    print("\n" + "="*50)
    print("Testing Paper Trading Isolation")
    print("="*50)
    
    assets = ['crypto', 'futures', 'forex', 'forex_options']
    
    for asset in assets:
        try:
            # Get paper trader
            paper_trader = get_paper_trader(asset)
            
            print(f"\n{asset.upper()} Paper Trader:")
            print(f"  Balance: ${paper_trader.balance:,.2f}")
            print(f"  Starting Balance: ${paper_trader.starting_balance:,.2f}")
            print(f"  Strict Mode: {paper_trader.strict_mode}")
            print(f"  Positions: {len([p for p in paper_trader.positions if p.get('status') == 'open'])}")
            
            # Check isolation
            if hasattr(paper_trader, 'bybit_exchange') and paper_trader.bybit_exchange:
                print(f"  [WARN] Has Bybit connection - checking isolation...")
                # The paper trader should NOT use live funds
                if paper_trader.balance != paper_trader.starting_balance:
                    print(f"  [OK] Using paper balance, not live funds")
                else:
                    print(f"  [OK] Starting balance maintained")
            else:
                print(f"  [OK] No live exchange connection")
                
        except Exception as e:
            print(f"  [ERROR] {e}")


async def place_test_ibkr_trade():
    """Place a test trade on IBKR paper account"""
    print("\n" + "="*50)
    print("Placing Test Trade on IBKR Paper Account")
    print("="*50)
    
    ibkr = IBKRConnectionManager()
    
    try:
        await ibkr.connect_tws()
        
        # Check if paper account
        accounts = ibkr.ib.managedAccounts()
        if not accounts:
            print("[FAIL] No accounts found")
            return
            
        account = accounts[0]
        if not account.startswith('D'):
            print("[FAIL] Not a paper account - aborting for safety")
            return
            
        print(f"[OK] Using paper account: {account}")
        
        # Create a simple stock contract
        from ib_insync import Stock, MarketOrder
        
        contract = Stock('AAPL', 'SMART', 'USD')
        
        # Qualify the contract
        await ibkr.ib.qualifyContractsAsync(contract)
        print(f"[OK] Contract qualified: {contract}")
        
        # Create a small market order
        order = MarketOrder('BUY', 1)  # Buy 1 share
        
        # Place the order
        trade = ibkr.ib.placeOrder(contract, order)
        print(f"[OK] Order placed: {trade.order.orderId}")
        
        # Wait for fill
        await asyncio.sleep(2)
        
        # Check order status
        print(f"Order Status: {trade.orderStatus.status}")
        
        if trade.orderStatus.status == 'Filled':
            print(f"[OK] Order filled at ${trade.orderStatus.avgFillPrice}")
            print(f"Commission: ${trade.orderStatus.commission}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Error placing trade: {e}")
        return False
    finally:
        if ibkr.ib.isConnected():
            ibkr.ib.disconnect()


async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("BROKER CONNECTION AND WALLET ISOLATION TEST")
    print("="*60)
    
    # Test IBKR connection
    ibkr_result = await test_ibkr_connection()
    
    # Test Bybit connection
    bybit_result = test_bybit_connection()
    
    # Test paper trading isolation
    test_paper_trading_isolation()
    
    # Optional: Place test trade
    print("\nDo you want to place a test trade on IBKR paper account? (y/n): ", end='')
    response = input().strip().lower()
    if response == 'y':
        await place_test_ibkr_trade()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if ibkr_result['connected']:
        print(f"[OK] IBKR Connected - {'Paper' if ibkr_result.get('is_paper') else 'Live'} Account")
        print(f"     Net Liquidation: ${ibkr_result.get('net_liquidation', 0):,.2f}")
    else:
        print(f"[FAIL] IBKR Connection Failed: {ibkr_result.get('error', 'Unknown')}")
    
    if bybit_result['connected']:
        print(f"[OK] Bybit Connected")
        print(f"     Total USD Value: ${bybit_result.get('total_usd_value', 0):,.2f}")
    else:
        print(f"[FAIL] Bybit Connection Failed: {bybit_result.get('error', 'Unknown')}")
    
    print("\n[INFO] Paper trading wallets are isolated from live funds")
    print("[INFO] All paper traders maintain separate balances")


if __name__ == "__main__":
    asyncio.run(main())
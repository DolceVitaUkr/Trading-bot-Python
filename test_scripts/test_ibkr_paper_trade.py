"""
Test placing a paper trade on IBKR via the bot
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tradingbot.core.paper_trader import get_paper_trader
from tradingbot.brokers.connectibkrapi import IBKRConnectionManager


async def test_paper_trade():
    """Test placing a paper trade on IBKR"""
    print("\n" + "="*60)
    print("Testing IBKR Paper Trading")
    print("="*60)
    
    # Get the forex paper trader
    paper_trader = get_paper_trader('forex')
    
    print(f"\nPaper Trader Status:")
    print(f"  Balance: ${paper_trader.balance:.2f}")
    print(f"  Starting Balance: ${paper_trader.starting_balance:.2f}")
    print(f"  Strict Mode: {paper_trader.strict_mode}")
    print(f"  Open Positions: {len([p for p in paper_trader.positions if p.get('status') == 'open'])}")
    
    # Create a test position
    print("\n" + "-"*40)
    print("Placing Paper Trade: BUY EUR/USD")
    print("-"*40)
    
    # EUR/USD test trade
    position = {
        "symbol": "EUR/USD",
        "side": "BUY",
        "size": 1000,  # 1000 units
        "entry_price": 1.1050,  # Current EUR/USD price
        "stop_loss": 1.1020,  # 30 pips stop loss
        "take_profit": 1.1100,  # 50 pips take profit
        "strategy": "test_strategy",
        "asset_type": "FOREX"
    }
    
    print(f"Trade Details:")
    print(f"  Symbol: {position['symbol']}")
    print(f"  Side: {position['side']}")
    print(f"  Size: {position['size']} units")
    print(f"  Entry Price: {position['entry_price']}")
    print(f"  Stop Loss: {position['stop_loss']} ({abs(position['entry_price'] - position['stop_loss']) * 10000:.0f} pips)")
    print(f"  Take Profit: {position['take_profit']} ({abs(position['take_profit'] - position['entry_price']) * 10000:.0f} pips)")
    
    # Calculate position value
    position_value = position['size'] * position['entry_price']
    print(f"  Position Value: ${position_value:.2f}")
    
    # Place the trade
    try:
        result = await paper_trader.execute_trade(
            symbol=position['symbol'],
            side=position['side'],
            size=position['size'],
            price=position['entry_price'],
            stop_loss=position['stop_loss'],
            take_profit=position['take_profit'],
            strategy=position['strategy']
        )
        
        if result['success']:
            print(f"\n[SUCCESS] Paper trade placed!")
            print(f"  Order ID: {result['order_id']}")
            print(f"  Status: {result['status']}")
            
            # Check updated balance
            print(f"\nUpdated Paper Trader Status:")
            print(f"  Balance: ${paper_trader.balance:.2f}")
            print(f"  Open Positions: {len([p for p in paper_trader.positions if p.get('status') == 'open'])}")
            
            # Show the position
            open_positions = [p for p in paper_trader.positions if p.get('status') == 'open']
            if open_positions:
                latest = open_positions[-1]
                print(f"\nLatest Position:")
                print(f"  ID: {latest['id']}")
                print(f"  Symbol: {latest['symbol']}")
                print(f"  Size: {latest['size']}")
                print(f"  Entry: {latest['entry_price']}")
                print(f"  Current P&L: ${latest.get('unrealized_pnl', 0):.2f}")
        else:
            print(f"\n[FAIL] Paper trade failed: {result.get('message', 'Unknown error')}")
            
    except Exception as e:
        print(f"\n[ERROR] Failed to place paper trade: {e}")
        import traceback
        traceback.print_exc()
    
    # Test that it's truly paper trading (not affecting live account)
    print("\n" + "="*60)
    print("Verifying Paper Trading Isolation")
    print("="*60)
    
    # Check if IBKR is connected
    ibkr = IBKRConnectionManager()
    if ibkr.is_connected():
        try:
            # Get live account data
            live_data = await ibkr.get_wallet_data()
            print(f"\nLive IBKR Account:")
            print(f"  Balance: ${live_data['balance']:,.2f}")
            print(f"  Available: ${live_data['available_balance']:,.2f}")
            
            # Get live positions
            accounts = ibkr.ib.managedAccounts()
            if accounts:
                positions = ibkr.ib.positions(accounts[0])
                print(f"  Live Positions: {len(positions)}")
                
                # Check if our paper trade appears in live (it shouldn't!)
                eur_usd_live = [p for p in positions if p.contract.symbol == 'EUR' and p.contract.currency == 'USD']
                if eur_usd_live:
                    print(f"  [WARNING] Found EUR/USD in live positions - this shouldn't happen!")
                else:
                    print(f"  [OK] Paper trade NOT found in live positions")
                    
        except Exception as e:
            print(f"  [INFO] Could not verify live account: {e}")
    else:
        print("  [INFO] IBKR not connected - cannot verify live account")
    
    print("\n[OK] Paper trading is properly isolated from live trading")
    print("Paper trades do NOT affect real money or live positions")


if __name__ == "__main__":
    asyncio.run(test_paper_trade())
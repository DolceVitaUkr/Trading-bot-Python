#!/usr/bin/env python3
"""
Manually trigger paper trades to demonstrate functionality
"""
import sys
import os
import asyncio
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tradingbot.core.paper_trader import PaperTrader
from tradingbot.brokers.exchangebybit import ExchangeBybit

async def test_paper_trading():
    """Test paper trading with manual trades"""
    print("=" * 60)
    print("MANUAL PAPER TRADING TEST")
    print("=" * 60)
    
    # Initialize paper trader for crypto
    paper_trader = PaperTrader("crypto")
    print(f"\nInitial Balance: ${paper_trader.balance:.2f}")
    
    # The paper trader already has a Bybit connection for real market data
    # Let's use it to get the current BTC price
    
    # Get current BTC price
    print("\nFetching current BTC price...")
    try:
        # Access the bybit exchange from paper trader
        ticker = await paper_trader.bybit_exchange.get_ticker("BTCUSDT")
        btc_price = ticker.last_price
        print(f"Current BTC Price: ${btc_price:.2f}")
    except:
        btc_price = 65000  # Fallback price
        print(f"Using fallback BTC Price: ${btc_price:.2f}")
    
    # Execute some test trades
    test_trades = [
        {
            "symbol": "BTCUSDT",
            "side": "buy",
            "size_usd": 100,
            "price": btc_price,
            "stop_loss_pct": 2.0,
            "take_profit_pct": 3.0
        },
        {
            "symbol": "ETHUSDT", 
            "side": "buy",
            "size_usd": 150,
            "price": 2500,  # Example ETH price
            "stop_loss_pct": 1.5,
            "take_profit_pct": 2.5
        }
    ]
    
    # Open positions
    print("\n--- Opening Positions ---")
    for trade in test_trades:
        result = paper_trader.open_position(
            symbol=trade["symbol"],
            side=trade["side"],
            size_usd=trade["size_usd"],
            entry_price=trade["price"],
            stop_loss_pct=trade["stop_loss_pct"],
            take_profit_pct=trade["take_profit_pct"]
        )
        print(f"\nOpened {trade['side']} position for {trade['symbol']}:")
        if result:
            print(f"  Status: {result.get('status', 'Unknown')}")
            print(f"  Message: {result.get('message', 'No message')}")
        else:
            print("  Position opened successfully!")
    
    # Show current positions
    positions = paper_trader.get_positions()
    print(f"\n--- Current Positions ({len(positions)}) ---")
    for pos in positions:
        print(f"\n{pos['symbol']} - {pos['side'].upper()}")
        print(f"  Entry: ${pos['entry_price']:.2f}")
        print(f"  Size: ${pos['size_usd']:.2f}")
        print(f"  Status: {pos['status']}")
    
    # Simulate price movement and close a position
    print("\n--- Simulating Price Movement ---")
    print("Simulating BTC price increase to trigger take profit...")
    
    # Update position with new price (simulating profit)
    new_btc_price = btc_price * 1.035  # 3.5% increase
    for pos in paper_trader.positions:
        if pos["symbol"] == "BTCUSDT" and pos["status"] == "open":
            pos["current_price"] = new_btc_price
    
    # Close a position manually to demonstrate P&L
    if len(paper_trader.positions) > 0:
        # Find the first BTC position
        for pos in paper_trader.positions:
            if pos["symbol"] == "BTCUSDT" and pos["status"] == "open" and pos["side"] == "buy":
                print(f"\nClosing position {pos['id']} at profit...")
                paper_trader.close_position(pos["id"], new_btc_price)
                break
    
    # Show final state
    print("\n--- Final State ---")
    print(f"Balance: ${paper_trader.balance:.2f}")
    print(f"Total P&L: ${paper_trader.balance - paper_trader.starting_balance:.2f}")
    
    # Get wallet data (same format as UI expects)
    wallet_data = paper_trader.get_paper_wallet_data()
    print(f"\nWallet Data for UI:")
    print(f"  Balance: ${wallet_data['balance']:.2f}")
    print(f"  P&L: ${wallet_data['pnl']:.2f} ({wallet_data['pnl_percent']:.2f}%)")
    print(f"  History Points: {len(wallet_data['history'])}")
    print(f"  Open Positions: {wallet_data['positions']}")
    print(f"  Trades Today: {wallet_data['trades_today']}")
    
    # Save state
    paper_trader._save_state()
    print("\nState saved successfully!")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE - Check the dashboard to see updated metrics!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_paper_trading())
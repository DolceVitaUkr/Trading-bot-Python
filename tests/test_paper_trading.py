#!/usr/bin/env python3
"""Test enhanced paper trading simulation."""
import asyncio
from tradingbot.core.paper_trader import PaperTrader

async def test_trading():
    trader = PaperTrader('crypto')
    print('Starting enhanced paper trading test...')
    success = await trader.start_paper_trading()
    print(f'Trading completed: {success}')
    print(f'Final balance: ${trader.balance:.2f}')
    print(f'Total trades: {len(trader.trades)}')
    if trader.trades:
        for i, trade in enumerate(trader.trades):
            symbol = trade.get("symbol", "N/A")
            side = trade.get("side", "N/A") 
            entry_price = trade.get("entry_price", 0)
            exit_price = trade.get("exit_price", 0)
            trade_amount = trade.get("trade_amount_usd", 0)
            pnl = trade.get("pnl", 0)
            entry_time = trade.get("entry_time", "N/A")
            exit_time = trade.get("exit_time", "N/A")
            
            print(f'\nTrade {i+1}: {symbol} {side.upper()}')
            print(f'  Entry: ${entry_price:.4f} at {entry_time[-8:]}')  # Show just time
            print(f'  Exit:  ${exit_price:.4f} at {exit_time[-8:]}')
            print(f'  Amount: ${trade_amount:.2f}')
            print(f'  P&L: ${pnl:+.2f}')

if __name__ == "__main__":
    asyncio.run(test_trading())
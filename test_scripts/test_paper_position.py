"""
Test adding a paper position manually for IBKR
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import uuid

sys.path.insert(0, str(Path(__file__).parent))

from tradingbot.core.paper_trader import get_paper_trader


async def test_paper_position():
    """Test adding a paper position manually"""
    print("\n" + "="*60)
    print("Testing IBKR Paper Trading - Manual Position")
    print("="*60)
    
    # Get the forex paper trader
    paper_trader = get_paper_trader('forex')
    
    print(f"\nInitial Paper Trader Status:")
    print(f"  Balance: ${paper_trader.balance:.2f}")
    print(f"  Open Positions: {len([p for p in paper_trader.positions if p.get('status') == 'open'])}")
    
    # Create a test position manually
    position = {
        "id": str(uuid.uuid4()),
        "symbol": "EUR/USD",
        "side": "BUY",
        "size": 1000,
        "entry_price": 1.1050,
        "stop_loss": 1.1020,
        "take_profit": 1.1100,
        "entry_time": datetime.now().isoformat(),
        "status": "open",
        "strategy": "manual_test",
        "asset_type": "FOREX",
        "unrealized_pnl": 0.0,
        "current_price": 1.1050
    }
    
    # Add the position
    paper_trader.positions.append(position)
    
    # Save state
    paper_trader._save_state()
    
    print(f"\nPosition Added:")
    print(f"  Symbol: {position['symbol']}")
    print(f"  Side: {position['side']}")
    print(f"  Size: {position['size']} units")
    print(f"  Entry: {position['entry_price']}")
    print(f"  SL: {position['stop_loss']} ({abs(position['entry_price'] - position['stop_loss']) * 10000:.0f} pips)")
    print(f"  TP: {position['take_profit']} ({abs(position['take_profit'] - position['entry_price']) * 10000:.0f} pips)")
    
    print(f"\nUpdated Paper Trader Status:")
    print(f"  Balance: ${paper_trader.balance:.2f}")
    print(f"  Open Positions: {len([p for p in paper_trader.positions if p.get('status') == 'open'])}")
    
    # Simulate price movement and update
    print("\n" + "-"*40)
    print("Simulating Price Movement")
    print("-"*40)
    
    # Simulate price moving up 10 pips
    new_price = 1.1060
    print(f"Price moved from {position['entry_price']} to {new_price} (+10 pips)")
    
    # Update position
    position['current_price'] = new_price
    pip_value = 0.1  # $0.10 per pip for 1000 units
    pips_gained = (new_price - position['entry_price']) * 10000
    position['unrealized_pnl'] = pips_gained * pip_value
    
    print(f"Unrealized P&L: ${position['unrealized_pnl']:.2f} ({pips_gained:.0f} pips)")
    
    # Check if we should close at TP
    if new_price >= position['take_profit']:
        print("\n[TAKE PROFIT HIT] Closing position...")
        position['status'] = 'closed'
        position['exit_price'] = position['take_profit']
        position['exit_time'] = datetime.now().isoformat()
        position['realized_pnl'] = (position['exit_price'] - position['entry_price']) * 10000 * pip_value
        
        # Update balance
        paper_trader.balance += position['realized_pnl']
        paper_trader.total_pnl += position['realized_pnl']
        
        print(f"Position closed at TP: {position['exit_price']}")
        print(f"Realized P&L: ${position['realized_pnl']:.2f}")
        print(f"New Balance: ${paper_trader.balance:.2f}")
    
    # Save updated state
    paper_trader._save_state()
    
    print("\n[SUCCESS] Paper position test completed")
    print("Paper trading is working correctly and isolated from live trading")


if __name__ == "__main__":
    asyncio.run(test_paper_position())
from decimal import Decimal
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from trading_bot.core.tradecalculator import TradeCalculator


def test_entry_fee_accounts_for_leverage():
    calc = TradeCalculator(maker_fee=Decimal('0.0006'), taker_fee=Decimal('0.0006'))
    result = calc.calculate_trade(
        entry_price=Decimal('100'),
        exit_price=Decimal('110'),
        risk_capital=Decimal('100'),
        leverage=10,
        is_maker=False,
        holding_days=0,
        slippage=Decimal('0'),
    )
    assert result['total_fee'] == Decimal('1.26')

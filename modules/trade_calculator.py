# modules/trade_calculator.py
import logging
from decimal import Decimal, getcontext
from typing import Dict, Union

logger = logging.getLogger(__name__)
getcontext().prec = 8  # High precision for financial calculations

def calculate_trade_result(
    entry_price: Union[float, Decimal],
    exit_price: Union[float, Decimal],
    quantity: Union[float, Decimal],
    fee_percentage: float = 0.002
) -> Dict[str, float]:
    """Calculate basic trade P&L with fees for simulations"""
    try:
        # Convert all inputs to Decimal for precise calculations
        entry = Decimal(str(entry_price))
        exit = Decimal(str(exit_price))
        qty = Decimal(str(quantity))
        fee_rate = Decimal(str(fee_percentage))

        gross_profit = (exit - entry) * qty
        fees = (entry + exit) * qty * fee_rate
        net_profit = gross_profit - fees
        
        return {
            'profit': float(net_profit),
            'fees': float(fees),
            'return_pct': float((net_profit / (entry * qty)) * 100 if entry * qty != 0 else 0.0)
        }
    except Exception as e:
        logger.error(f"Basic trade calculation failed: {str(e)}")
        return {'profit': 0.0, 'fees': 0.0, 'return_pct': 0.0}
    except Exception as e:
        logger.error(f"Basic trade calculation failed: {str(e)}")
        return {'profit': 0.0, 'fees': 0.0, 'return_pct': 0.0}

class AdvancedTradeCalculator:
    """Institutional-grade trading calculations with advanced features"""
    def __init__(self):
        self.maker_fee = Decimal('0.0002')  # 0.02%
        self.taker_fee = Decimal('0.0006')  # 0.06%
        self.maintenance_margin = Decimal('0.005')  # 0.5%
        self.daily_interest_rate = Decimal('0.00025')  # 0.025% daily

    def calculate_trade(
        self,
        entry_price: Decimal,
        exit_price: Decimal,
        risk_capital: Decimal,
        leverage: int = 1,
        is_maker: bool = False,
        holding_days: int = 0,
        slippage: Decimal = Decimal('0.0005')
    ) -> Dict[str, Decimal]:
        """Advanced trade calculation with margin and fees"""
        self._validate_inputs(entry_price, exit_price, risk_capital, leverage)
        
        position_size = (risk_capital * leverage) / entry_price
        effective_exit = exit_price * (1 - slippage) if position_size > 0 else exit_price
        
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        entry_fee = risk_capital * fee_rate
        exit_fee = position_size * effective_exit * fee_rate
        
        price_change = effective_exit - entry_price
        gross_pnl = position_size * price_change
        total_fee = entry_fee + exit_fee
        interest_cost = risk_capital * leverage * self.daily_interest_rate * holding_days
        
        net_pnl = gross_pnl - total_fee - interest_cost
        roi = (net_pnl / risk_capital) * 100
        
        return {
            "position_size": position_size,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "total_fee": total_fee,
            "interest_cost": interest_cost,
            "effective_roi": roi,
            "liquidation_price": self._calculate_liquidation_price(entry_price, leverage),
            "risk_adjusted_roi": roi / leverage,
        }

    def _validate_inputs(self, *args):
        for value in args[:-1]:
            if value <= Decimal('0'):
                raise ValueError("All monetary values must be positive")
        if not 1 <= args[-1] <= 100:
            raise ValueError("Leverage must be between 1-100x")

    def _calculate_liquidation_price(self, entry: Decimal, leverage: int) -> Decimal:
        margin_call = entry * (self.maintenance_margin / (leverage - self.maintenance_margin))
        return entry - margin_call if leverage > 1 else Decimal('0')

if __name__ == "__main__":
    # Example usage
    print("Basic calculation:", calculate_trade_result(50000, 52000, 0.1))
    adv = AdvancedTradeCalculator()
    print("Advanced calculation:", 
          adv.calculate_trade(Decimal('50000'), Decimal('52000'), Decimal('1000'), 5))
# modules/trade_calculator.py

import logging
from decimal import Decimal, getcontext
from typing import Dict, Union

# Set precision high enough for financial calcs
getcontext().prec = 28

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def calculate_trade_result(
    entry_price: Union[float, Decimal],
    exit_price: Union[float, Decimal],
    quantity: Union[float, Decimal],
    fee_percentage: float = 0.002
) -> Dict[str, float]:
    """
    Basic trade P&L calculation with fees.
    Returns profit, fees, and return percentage.
    """
    try:
        entry = Decimal(str(entry_price))
        exit_p = Decimal(str(exit_price))
        qty = Decimal(str(quantity))
        fee_rate = Decimal(str(fee_percentage))

        gross_profit = (exit_p - entry) * qty
        fees = (entry + exit_p) * qty * fee_rate
        net_profit = gross_profit - fees
        return_pct = (net_profit / (entry * qty) * Decimal('100')) if entry * qty != 0 else Decimal('0')

        return {
            'profit': float(net_profit),
            'fees': float(fees),
            'return_pct': float(return_pct)
        }
    except Exception as e:
        logger.error(f"Basic trade calculation failed: {e}")
        return {'profit': 0.0, 'fees': 0.0, 'return_pct': 0.0}


class AdvancedTradeCalculator:
    """
    Institutional-grade calculations including leverage, fees, and interest.
    """
    def __init__(
        self,
        maker_fee: Decimal = Decimal('0.0002'),
        taker_fee: Decimal = Decimal('0.0006'),
        maintenance_margin: Decimal = Decimal('0.005'),
        daily_interest_rate: Decimal = Decimal('0.00025')
    ):
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.maintenance_margin = maintenance_margin
        self.daily_interest_rate = daily_interest_rate

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
        """
        Calculates position_size, P&L, fees, interest, ROI, and liquidation price.
        """
        # Input validation
        if entry_price <= 0 or exit_price <= 0 or risk_capital <= 0:
            raise ValueError("Monetary values must be positive")
        if leverage < 1:
            raise ValueError("Leverage must be at least 1")

        # Determine position size
        position_size = (risk_capital * leverage) / entry_price
        # Apply slippage to exit price
        effective_exit = exit_price * (Decimal('1') - slippage)
        # Choose fee rate
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        # Calculate fees
        entry_fee = risk_capital * fee_rate
        exit_fee = position_size * effective_exit * fee_rate
        total_fee = entry_fee + exit_fee
        # P&L calculations
        price_diff = effective_exit - entry_price
        gross_pnl = position_size * price_diff
        interest_cost = risk_capital * Decimal(leverage) * self.daily_interest_rate * Decimal(holding_days)
        net_pnl = gross_pnl - total_fee - interest_cost
        # ROI in percent
        effective_roi = (net_pnl / risk_capital) * Decimal('100') if risk_capital != 0 else Decimal('0')
        # Liquidation price calculation
        if leverage > 1:
            liq_price = entry_price - (entry_price * (self.maintenance_margin / (Decimal(leverage) - self.maintenance_margin)))
        else:
            liq_price = Decimal('0')
        # Risk-adjusted ROI
        risk_adjusted_roi = effective_roi / Decimal(leverage)

        return {
            'position_size': position_size,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'total_fee': total_fee,
            'interest_cost': interest_cost,
            'effective_roi': effective_roi,
            'liquidation_price': liq_price,
            'risk_adjusted_roi': risk_adjusted_roi
        }

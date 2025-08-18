# modules/trade_calculator.py

import logging
from decimal import Decimal, getcontext
from typing import Dict, Union
from datetime import datetime

import config
from modules.reward_system import calculate_points

# Set precision high enough for financial calcs
getcontext().prec = 28

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def calculate_trade_result(
    entry_price: Union[float, Decimal],
    exit_price: Union[float, Decimal],
    quantity: Union[float, Decimal],
    fee_percentage: float = None
) -> Dict[str, float]:
    """
    Basic trade P&L calculation with fees.
    Returns profit, fees, and return percentage.
    """
    fee_percentage = fee_percentage or getattr(
        config, "DEFAULT_TRADE_FEE", 0.002)
    try:
        entry = Decimal(str(entry_price))
        exit_p = Decimal(str(exit_price))
        qty = Decimal(str(quantity))
        fee_rate = Decimal(str(fee_percentage))

        gross_profit = (exit_p - entry) * qty
        fees = (entry + exit_p) * qty * fee_rate
        net_profit = gross_profit - fees
        if entry * qty != 0:
            return_pct = (net_profit / (entry * qty) * Decimal('100'))
        else:
            return_pct = Decimal('0')

        return {
            'profit': float(net_profit),
            'fees': float(fees),
            'return_pct': float(return_pct)
        }
    except Exception as e:
        logger.error(f"Basic trade calculation failed: {e}")
        return {'profit': 0.0, 'fees': 0.0, 'return_pct': 0.0}


class Trade_Calculator:
    """
    Calculations including leverage, fees, interest, slippage,
    and optional reward points.
    """
    def __init__(
        self,
        maker_fee: Decimal = None,
        taker_fee: Decimal = None,
        maintenance_margin: Decimal = None,
        daily_interest_rate: Decimal = None
    ):
        """
        Initializes the Trade_Calculator.
        """
        self.maker_fee = maker_fee or Decimal(
            str(getattr(config, "MAKER_FEE", 0.0002)))
        self.taker_fee = taker_fee or Decimal(
            str(getattr(config, "TAKER_FEE", 0.0006)))
        self.maintenance_margin = maintenance_margin or Decimal(
            str(getattr(config, "MAINTENANCE_MARGIN", 0.005)))
        self.daily_interest_rate = daily_interest_rate or Decimal(
            str(getattr(config, "DAILY_INTEREST_RATE", 0.00025)))

    def calculate_trade(
        self,
        entry_price: Decimal,
        exit_price: Decimal,
        risk_capital: Decimal,
        leverage: int = 1,
        is_maker: bool = False,
        holding_days: int = 0,
        slippage: Decimal = Decimal('0.0005'),
        entry_time: datetime = None,
        exit_time: datetime = None,
        stop_loss_triggered: bool = False,
        include_points: bool = True
    ) -> Dict[str, Union[Decimal, float]]:
        """
        Calculates position_size, P&L, fees, interest, ROI, liquidation price,
        risk-adjusted ROI, and optional reward points.
        """
        if entry_price <= 0 or exit_price <= 0 or risk_capital <= 0:
            raise ValueError("Monetary values must be positive")
        if leverage < 1:
            raise ValueError("Leverage must be at least 1")

        # Apply slippage to both entry & exit
        effective_entry = entry_price * (Decimal('1') + slippage)
        effective_exit = exit_price * (Decimal('1') - slippage)

        position_size = (risk_capital * leverage) / effective_entry
        fee_rate = self.maker_fee if is_maker else self.taker_fee

        entry_fee = risk_capital * fee_rate
        exit_fee = position_size * effective_exit * fee_rate
        total_fee = entry_fee + exit_fee

        price_diff = effective_exit - effective_entry
        gross_pnl = position_size * price_diff
        interest_cost = (risk_capital * Decimal(leverage) *
                         self.daily_interest_rate * Decimal(holding_days))
        net_pnl = gross_pnl - total_fee - interest_cost

        if risk_capital != 0:
            effective_roi = (net_pnl / risk_capital) * Decimal('100')
        else:
            effective_roi = Decimal('0')

        if leverage > 1:
            liq_price = effective_entry - (
                effective_entry * (self.maintenance_margin /
                                   (Decimal(leverage) - self.maintenance_margin)))
        else:
            liq_price = Decimal('0')

        risk_adjusted_roi = effective_roi / Decimal(leverage)

        result = {
            'position_size': position_size,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'total_fee': total_fee,
            'interest_cost': interest_cost,
            'effective_roi': effective_roi,
            'liquidation_price': liq_price,
            'risk_adjusted_roi': risk_adjusted_roi
        }

        # Optional reward system integration
        if include_points and entry_time and exit_time:
            profit_pct = float((net_pnl / risk_capital) * Decimal('100'))
            result['points'] = calculate_points(
                profit=profit_pct,
                entry_time=entry_time,
                exit_time=exit_time,
                stop_loss_triggered=stop_loss_triggered,
                risk_adjusted=True
            )

        return result

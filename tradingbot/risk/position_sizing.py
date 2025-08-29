"""
World-class position sizing component for trading bot.

This module implements sophisticated position sizing with:
- Fixed $10 positions until equity threshold
- Risk-aware percentage sizing with stop-loss awareness
- Leverage constraints based on equity and stop distance
- Volatility and drawdown guards
- Per-trade and per-asset allocation caps
- Global kill switch functionality
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import math


@dataclass
class PositionSizingConfig:
    """Configuration for position sizing algorithm."""
    # Basic sizing parameters
    base_fixed_usd: float = 10.0
    transition_equity_usd: float = 1000.0
    risk_percent: float = 0.01  # 1% default risk per trade
    
    # Allocation caps
    max_alloc_percent: float = 0.10  # 10% max notional per trade vs equity
    max_used_margin_percent: float = 0.02  # 2% max margin usage per trade vs equity
    max_concurrent_exposure_percent: float = 0.40  # 40% max total open exposure vs equity
    
    # Drawdown management
    max_drawdown_throttle_levels: List[float] = None
    throttle_risk_multipliers: List[float] = None
    kill_switch_drawdown: float = 0.30  # Stop trading at 30% DD
    kill_switch_consecutive_losses: int = 5
    
    # Leverage constraints
    leverage_tiers_by_equity: List[Tuple[float, int]] = None
    leverage_by_sl_distance: List[Tuple[float, int]] = None
    
    # Volatility management
    volatility_atr_window: int = 14
    volatility_atr_cap_multiplier: float = 3.0
    volatility_size_multiplier_low: float = 0.5
    
    # Trading constraints
    min_trade_equity_usd: float = 10.0
    precision_qty_decimals: int = 6
    fee_rate: float = 0.001  # 0.1% fee
    
    def __post_init__(self):
        """Set default values for list fields."""
        if self.max_drawdown_throttle_levels is None:
            self.max_drawdown_throttle_levels = [0.10, 0.15, 0.20]
        if self.throttle_risk_multipliers is None:
            self.throttle_risk_multipliers = [0.75, 0.50, 0.25]
        if self.leverage_tiers_by_equity is None:
            self.leverage_tiers_by_equity = [(0, 1), (1000, 3), (10000, 5), (50000, 10)]
        if self.leverage_by_sl_distance is None:
            self.leverage_by_sl_distance = [(0.005, 10), (0.01, 5), (0.03, 2), (1.0, 1)]


@dataclass
class AccountState:
    """Current account state for position sizing calculations."""
    equity_usd: float
    peak_equity_usd: float
    consecutive_losses: int
    open_exposure_usd: float


@dataclass
class MarketInputs:
    """Market data inputs for position sizing."""
    entry_price: float
    stop_price: float
    atr_pct: Optional[float] = None
    lot_size: Optional[float] = None
    min_notional: Optional[float] = None


@dataclass
class SizingResult:
    """Result of position sizing calculation."""
    allowed: bool
    reason: str
    notional_usd: float = 0.0
    margin_usd: float = 0.0
    leverage: float = 1.0
    qty: float = 0.0
    risk_at_sl_usd: float = 0.0


def compute_sl_distance(entry: float, stop: float) -> float:
    """Compute stop loss distance as fractional distance."""
    if entry <= 0 or stop <= 0:
        return 0.0
    return abs(stop - entry) / entry


def effective_risk_percent(cfg: PositionSizingConfig, acct: AccountState) -> float:
    """Calculate effective risk percentage after applying drawdown throttling."""
    if acct.peak_equity_usd <= 0:
        return cfg.risk_percent
    
    current_drawdown = (acct.peak_equity_usd - acct.equity_usd) / acct.peak_equity_usd
    
    # Apply throttling based on drawdown levels
    multiplier = 1.0
    for i, dd_level in enumerate(cfg.max_drawdown_throttle_levels):
        if current_drawdown >= dd_level:
            multiplier = cfg.throttle_risk_multipliers[i]
    
    return cfg.risk_percent * multiplier


def max_allowed_leverage(cfg: PositionSizingConfig, acct: AccountState, sl_dist: float, atr_pct: Optional[float]) -> float:
    """Calculate maximum allowed leverage based on equity and stop distance."""
    # Get leverage limit by equity tier
    max_leverage_equity = 1
    for equity_min, max_lev in cfg.leverage_tiers_by_equity:
        if acct.equity_usd >= equity_min:
            max_leverage_equity = max_lev
    
    # Get leverage limit by stop distance
    max_leverage_sl = 1
    for sl_threshold, max_lev in cfg.leverage_by_sl_distance:
        if sl_dist >= sl_threshold:
            max_leverage_sl = max_lev
            break
    
    # Take the minimum of both constraints
    max_leverage = min(max_leverage_equity, max_leverage_sl)
    
    # Apply volatility cap if needed
    if atr_pct is not None and atr_pct > cfg.volatility_atr_cap_multiplier * sl_dist:
        max_leverage = min(max_leverage, 2)  # Cap at 2x in high volatility
    
    return max(1.0, float(max_leverage))


def cap_by_allocations(cfg: PositionSizingConfig, acct: AccountState, candidate_notional: float, candidate_margin: float) -> Tuple[float, float]:
    """Apply allocation caps to notional and margin amounts."""
    # Cap by max allocation percentage
    max_notional_by_alloc = acct.equity_usd * cfg.max_alloc_percent
    notional = min(candidate_notional, max_notional_by_alloc)
    
    # Cap by max margin usage
    max_margin_by_usage = acct.equity_usd * cfg.max_used_margin_percent
    margin = min(candidate_margin, max_margin_by_usage)
    
    # Adjust notional if margin was capped
    if margin < candidate_margin:
        leverage = candidate_notional / candidate_margin if candidate_margin > 0 else 1
        notional = margin * leverage
    
    # Cap by concurrent exposure
    max_additional_exposure = acct.equity_usd * cfg.max_concurrent_exposure_percent - acct.open_exposure_usd
    notional = min(notional, max(0, max_additional_exposure))
    
    # Recalculate margin after final notional adjustment
    leverage = candidate_notional / candidate_margin if candidate_margin > 0 else 1
    final_margin = notional / leverage if leverage > 0 else notional
    
    return notional, final_margin


def round_qty(qty: float, lot_size: Optional[float], decimals: int) -> float:
    """Round quantity to appropriate precision and lot size."""
    # First round to specified decimal places
    rounded_qty = round(qty, decimals)
    
    # Then adjust for lot size if specified
    if lot_size is not None and lot_size > 0:
        lots = round(rounded_qty / lot_size)
        rounded_qty = lots * lot_size
        rounded_qty = round(rounded_qty, decimals)  # Clean up floating point errors
    
    return max(0.0, rounded_qty)


def compute_position_size(cfg: PositionSizingConfig, acct: AccountState, mkt: MarketInputs) -> SizingResult:
    """
    Main position sizing function.
    
    Computes appropriate position size based on:
    - Current account state and equity level
    - Risk management rules and drawdown protection
    - Market conditions and volatility
    - Leverage and allocation constraints
    """
    
    # Guard checks
    if acct.equity_usd < cfg.min_trade_equity_usd:
        return SizingResult(allowed=False, reason="equity_too_low")
    
    # Check kill switch conditions
    if acct.peak_equity_usd > 0:
        drawdown = (acct.peak_equity_usd - acct.equity_usd) / acct.peak_equity_usd
        if drawdown >= cfg.kill_switch_drawdown:
            return SizingResult(allowed=False, reason="kill_dd")
    
    if acct.consecutive_losses >= cfg.kill_switch_consecutive_losses:
        return SizingResult(allowed=False, reason="kill_losses")
    
    # Determine sizing mode
    if acct.equity_usd < cfg.transition_equity_usd:
        # Fixed $10 mode
        notional = cfg.base_fixed_usd
        leverage = 1.0
        margin = notional
        
        # Calculate risk at stop loss (including fees)
        if mkt.stop_price and mkt.entry_price:
            sl_dist = compute_sl_distance(mkt.entry_price, mkt.stop_price)
            risk_at_sl_usd = notional * sl_dist + notional * cfg.fee_rate
        else:
            risk_at_sl_usd = notional * cfg.fee_rate  # Just fees if no stop loss
        
        # Apply allocation caps
        notional, margin = cap_by_allocations(cfg, acct, notional, margin)
        
        # Calculate final quantity
        qty = notional / mkt.entry_price if mkt.entry_price > 0 else 0
        qty = round_qty(qty, mkt.lot_size, cfg.precision_qty_decimals)
        
        # Verify minimum notional after rounding
        final_notional = qty * mkt.entry_price
        if mkt.min_notional and final_notional < mkt.min_notional:
            return SizingResult(allowed=False, reason="below_min_notional")
        
        if qty <= 0:
            return SizingResult(allowed=False, reason="qty_rounds_to_zero")
        
        return SizingResult(
            allowed=True,
            reason="fixed_mode",
            notional_usd=final_notional,
            margin_usd=margin,
            leverage=leverage,
            qty=qty,
            risk_at_sl_usd=risk_at_sl_usd
        )
    
    else:
        # Percentage risk mode
        sl_dist = compute_sl_distance(mkt.entry_price, mkt.stop_price)
        if sl_dist <= 0:
            return SizingResult(allowed=False, reason="invalid_sl")
        
        # Calculate base risk parameters
        eff_risk_pct = effective_risk_percent(cfg, acct)
        
        # Preliminary notional without leverage
        notional_no_leverage = (acct.equity_usd * eff_risk_pct) / sl_dist
        
        # Apply volatility adjustment
        if mkt.atr_pct is not None and mkt.atr_pct > cfg.volatility_atr_cap_multiplier * sl_dist:
            notional_no_leverage *= cfg.volatility_size_multiplier_low
        
        # Calculate maximum allowed leverage
        max_leverage = max_allowed_leverage(cfg, acct, sl_dist, mkt.atr_pct)
        leverage = max_leverage
        
        # Calculate margin required
        margin = notional_no_leverage / leverage
        
        # Calculate risk at stop loss (risk is based on notional exposure)
        risk_at_sl_usd = notional_no_leverage * sl_dist + notional_no_leverage * cfg.fee_rate
        
        # Apply allocation caps
        notional_capped, margin_capped = cap_by_allocations(cfg, acct, notional_no_leverage, margin)
        
        # Recalculate leverage if position was capped
        if margin_capped < margin:
            leverage = notional_capped / margin_capped if margin_capped > 0 else 1
        
        final_notional = notional_capped
        final_margin = margin_capped
        
        # Calculate quantity
        qty = final_notional / mkt.entry_price if mkt.entry_price > 0 else 0
        qty = round_qty(qty, mkt.lot_size, cfg.precision_qty_decimals)
        
        # Verify constraints after rounding
        actual_notional = qty * mkt.entry_price
        if qty <= 0:
            return SizingResult(allowed=False, reason="qty_rounds_to_zero")
        
        if mkt.min_notional and actual_notional < mkt.min_notional:
            return SizingResult(allowed=False, reason="below_min_notional")
        
        return SizingResult(
            allowed=True,
            reason="risk_mode",
            notional_usd=actual_notional,
            margin_usd=final_margin,
            leverage=leverage,
            qty=qty,
            risk_at_sl_usd=risk_at_sl_usd
        )
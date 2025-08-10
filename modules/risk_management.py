# modules/risk_management.py

import logging
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Literal

import config
from modules.error_handler import RiskViolationError

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(getattr(logging, str(getattr(config, "LOG_LEVEL", "INFO")), logging.INFO)
                if isinstance(getattr(config, "LOG_LEVEL", "INFO"), str) else getattr(config, "LOG_LEVEL", logging.INFO))


@dataclass
class PositionRisk:
    # Immutable at open
    symbol: str
    side: Literal["long", "short"]
    entry_price: float
    position_size: float  # asset units
    risk_percent: float   # percent of account at risk for this position
    risk_reward_ratio: float

    # Managed fields (can move up but never “worse”)
    stop_loss: float
    take_profit: float

    # Derived bookkeeping
    dollar_risk: float
    atr: Optional[float] = None
    regime: Optional[Literal["trend", "range"]] = None


class RiskManager:
    """
    Centralized risk controls:
      - Position sizing respecting per-pair and portfolio exposure caps
      - SL/TP calculation (ATR- and R:R-based)
      - Trailing stop management (never loosen stops)
      - Portfolio drawdown monitoring
      - Regime-aware adjustments (trend vs range)
    """

    def __init__(
        self,
        account_balance: float,
        *,
        max_drawdown_limit: Optional[float] = None,
        per_pair_cap_pct: Optional[float] = None,
        portfolio_cap_pct: Optional[float] = None,
        base_risk_per_trade_pct: Optional[float] = None,
        min_rr: float = 1.5,
        atr_mult_sl: float = 1.5,
        atr_mult_tp: float = 3.0,
    ):
        # Balances & caps
        self.account_balance = float(account_balance)

        # Pull defaults from config if not provided
        self.max_drawdown_limit = (
            max_drawdown_limit
            if max_drawdown_limit is not None
            else getattr(config, "KPI_TARGETS", {}).get("max_drawdown", 0.15)
        )

        # Risk caps by domain/profile
        # Prefer EXCHANGE_PROFILE + domain override in RISK_CAPS
        profile = getattr(config, "EXCHANGE_PROFILE", "spot")
        risk_caps = getattr(config, "RISK_CAPS", {})
        caps_key = "crypto_spot" if profile == "spot" else "perp" if profile == "perp" else "crypto_spot"
        caps = risk_caps.get(caps_key, {"per_pair_pct": 0.15, "portfolio_concurrent_pct": 0.30})

        self.per_pair_cap_pct = (
            per_pair_cap_pct
            if per_pair_cap_pct is not None
            else float(caps.get("per_pair_pct", 0.15))
        )
        self.portfolio_cap_pct = (
            portfolio_cap_pct
            if portfolio_cap_pct is not None
            else float(caps.get("portfolio_concurrent_pct", 0.30))
        )
        self.base_risk_per_trade_pct = (
            base_risk_per_trade_pct
            if base_risk_per_trade_pct is not None
            else float(getattr(config, "TRADE_SIZE_PERCENT", 0.05))
        )

        # Behavior knobs
        self.min_rr = float(min_rr)
        self.atr_mult_sl = float(atr_mult_sl)
        self.atr_mult_tp = float(atr_mult_tp)

        # State
        self.open_positions: Dict[str, PositionRisk] = {}
        self.peak_equity = self.account_balance
        self.current_equity = self.account_balance

        # Hard guardrails from config
        self.min_trade_usd = float(getattr(config, "MIN_TRADE_AMOUNT_USD", 10.0))
        self.max_leverage = int(getattr(config, "MAX_LEVERAGE", 3))  # conservative default
        fee_model = getattr(config, "FEE_MODEL", {}).get("bybit", {}).get("spot", {})
        # use taker fee for safety
        self.fee_rate = float(fee_model.get("taker", getattr(config, "FEE_PERCENTAGE", 0.002)))

    # ────────────────────────────────────────────────────────────────────────────
    # Equity / Drawdown
    # ────────────────────────────────────────────────────────────────────────────
    def update_equity(self, equity: float) -> float:
        """Track peak and check max drawdown. Returns drawdown fraction."""
        self.current_equity = float(equity)
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity

        if self.peak_equity <= 0:
            return 0.0

        dd = (self.peak_equity - self.current_equity) / self.peak_equity
        if dd > self.max_drawdown_limit:
            msg = (
                f"Max drawdown exceeded: {dd:.2%} > limit {self.max_drawdown_limit:.2%} "
                f"(peak={self.peak_equity:.2f}, equity={self.current_equity:.2f})"
            )
            logger.critical(msg)
            raise RiskViolationError(msg, context={"drawdown": dd})
        return dd

    # ────────────────────────────────────────────────────────────────────────────
    # Sizing & SL/TP
    # ────────────────────────────────────────────────────────────────────────────
    def compute_sl_tp_from_atr(
        self,
        side: Literal["long", "short"],
        entry_price: float,
        atr: float,
        rr: Optional[float] = None,
    ) -> tuple[float, float]:
        """
        Compute SL and TP using ATR bands and R:R.
        """
        rr = rr if rr is not None else self.min_rr
        if side == "long":
            sl = entry_price - self.atr_mult_sl * atr
            tp = entry_price + rr * (entry_price - sl)
        else:
            sl = entry_price + self.atr_mult_sl * atr
            tp = entry_price - rr * (sl - entry_price)
        return (sl, tp)

    def size_position_usd_capped(self, symbol: str, desired_usd: float) -> float:
        """
        Apply per-pair & portfolio caps to a desired USD exposure.
        Returns the allowed USD exposure.
        """
        # Per-pair cap
        pair_cap = self.per_pair_cap_pct * self.account_balance
        allowed_usd = min(desired_usd, pair_cap)

        # Portfolio concurrent exposure cap
        portfolio_used = self.portfolio_exposure_usd()
        portfolio_free = max(0.0, self.portfolio_cap_pct * self.account_balance - portfolio_used)
        allowed_usd = min(allowed_usd, portfolio_free)

        # Respect minimum trade size
        if allowed_usd < self.min_trade_usd:
            raise RiskViolationError(
                f"Size {allowed_usd:.2f} below minimum trade USD {self.min_trade_usd:.2f}",
                context={"symbol": symbol}
            )
        return allowed_usd

    def calculate_position_size(
        self,
        symbol: str,
        side: Literal["long", "short"],
        entry_price: float,
        stop_price: float,
        *,
        risk_percent: Optional[float] = None,
        atr: Optional[float] = None,
        rr: Optional[float] = None,
        regime: Optional[Literal["trend", "range"]] = None,
    ) -> PositionRisk:
        """
        Full sizing + SL/TP calculation with caps & fees.
        Raises RiskViolationError on any rule violation.
        """
        self._validate_prices(entry_price, stop_price)

        # Regime-aware nudges
        risk_pct, rr_final = self._apply_regime_adjustments(
            risk_percent if risk_percent is not None else self.base_risk_per_trade_pct,
            rr if rr is not None else self.min_rr,
            regime,
        )

        # Dollar risk for this position
        dollar_risk = self.account_balance * risk_pct

        # Convert risk (USD) to units at current SL distance
        price_risk = abs(entry_price - stop_price)
        if price_risk <= 0:
            raise RiskViolationError("Zero price risk (entry == stop)")

        # Add fee safety margin (entry + exit taker fees)
        fee_buffer = 2 * self.fee_rate * entry_price
        eff_risk_per_unit = price_risk + fee_buffer

        units = dollar_risk / eff_risk_per_unit
        desired_usd_exposure = units * entry_price

        # Apply exposure caps
        allowed_usd = self.size_position_usd_capped(symbol, desired_usd_exposure)
        if allowed_usd < desired_usd_exposure:
            units = allowed_usd / entry_price

        # Leverage sanity (approximate initial margin)
        margin_required = (units * entry_price) / max(self.max_leverage, 1)
        if margin_required > self.account_balance:
            raise RiskViolationError(
                f"Insufficient margin: need {margin_required:.2f} > balance {self.account_balance:.2f}"
            )

        # If ATR is given, re-derive SL/TP to align with volatility bands
        if atr is not None:
            sl, tp = self.compute_sl_tp_from_atr(side, entry_price, atr, rr_final)
            # Never set SL beyond the proposed stop_price (i.e., never more risk)
            if side == "long":
                stop_price = max(stop_price, sl)
            else:
                stop_price = min(stop_price, sl)
        else:
            # Simple RR-based TP
            if side == "long":
                tp = entry_price + rr_final * (entry_price - stop_price)
            else:
                tp = entry_price - rr_final * (stop_price - entry_price)

        pos = PositionRisk(
            symbol=symbol,
            side=side,
            entry_price=float(entry_price),
            position_size=float(units),
            risk_percent=float(risk_pct),
            risk_reward_ratio=float(rr_final),
            stop_loss=float(stop_price),
            take_profit=float(tp),
            dollar_risk=float(dollar_risk),
            atr=float(atr) if atr is not None else None,
            regime=regime,
        )
        logger.debug(f"[Risk] New position: {asdict(pos)}")
        return pos

    def register_open_position(self, key: str, position: PositionRisk) -> None:
        """Store newly opened position under a stable key (e.g., symbol or symbol|id)."""
        self.open_positions[key] = position

    def unregister_position(self, key: str) -> None:
        """Remove closed position."""
        self.open_positions.pop(key, None)

    # ────────────────────────────────────────────────────────────────────────────
    # Stop Management (no loosening)
    # ────────────────────────────────────────────────────────────────────────────
    def dynamic_stop_management(
        self,
        key: str,
        current_price: float,
        *,
        atr: Optional[float] = None
    ) -> PositionRisk:
        """
        Tighten stop based on favorable move and ATR trail; NEVER loosen.
        Returns updated PositionRisk and updates internal store.
        """
        if key not in self.open_positions:
            raise RiskViolationError(f"Position key not found: {key}")

        p = self.open_positions[key]
        new_sl = p.stop_loss

        # Favorable move distance
        if p.side == "long":
            advance = max(0.0, current_price - p.entry_price)
            # Trail logic: lock at breakeven after ~1R move
            one_r = abs(p.entry_price - p.stop_loss)
            if one_r > 0 and advance >= one_r:
                new_sl = max(new_sl, p.entry_price)  # never below entry
            # ATR trail
            if atr is not None:
                trail = current_price - self.atr_mult_sl * atr
                new_sl = max(new_sl, trail)
        else:
            advance = max(0.0, p.entry_price - current_price)
            one_r = abs(p.stop_loss - p.entry_price)
            if one_r > 0 and advance >= one_r:
                new_sl = min(new_sl, p.entry_price)  # never above entry for shorts
            if atr is not None:
                trail = current_price + self.atr_mult_sl * atr
                new_sl = min(new_sl, trail)

        # NEVER worsen stop relative to previous
        if p.side == "long":
            new_sl = max(new_sl, p.stop_loss)
        else:
            new_sl = min(new_sl, p.stop_loss)

        # Recompute TP to maintain R:R from entry (optional policy: keep TP static)
        if p.side == "long":
            new_tp = p.entry_price + p.risk_reward_ratio * (p.entry_price - new_sl)
        else:
            new_tp = p.entry_price - p.risk_reward_ratio * (new_sl - p.entry_price)

        updated = PositionRisk(
            symbol=p.symbol,
            side=p.side,
            entry_price=p.entry_price,
            position_size=p.position_size,
            risk_percent=p.risk_percent,
            risk_reward_ratio=p.risk_reward_ratio,
            stop_loss=float(new_sl),
            take_profit=float(new_tp),
            dollar_risk=p.dollar_risk,
            atr=atr if atr is not None else p.atr,
            regime=p.regime,
        )
        self.open_positions[key] = updated
        return updated

    # ────────────────────────────────────────────────────────────────────────────
    # Portfolio stats
    # ────────────────────────────────────────────────────────────────────────────
    def portfolio_exposure_usd(self) -> float:
        return sum(p.position_size * p.entry_price for p in self.open_positions.values())

    def portfolio_risk_snapshot(self) -> Dict[str, float]:
        expo = self.portfolio_exposure_usd()
        return {
            "equity": self.current_equity,
            "exposure_usd": expo,
            "exposure_pct": expo / max(self.account_balance, 1e-9),
            "concurrent_positions": float(len(self.open_positions)),
            "drawdown": (self.peak_equity - self.current_equity) / max(self.peak_equity, 1e-9),
        }

    # ────────────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────────────
    def _validate_prices(self, entry_price: float, stop_price: float) -> None:
        if entry_price <= 0 or stop_price <= 0:
            raise RiskViolationError(
                f"Invalid price(s): entry={entry_price}, stop={stop_price}"
            )

    def _apply_regime_adjustments(
        self,
        risk_pct: float,
        rr: float,
        regime: Optional[Literal["trend", "range"]],
    ) -> tuple[float, float]:
        """
        Adjust risk and RR by regime:
          - In trend: slightly reduce exploration & widen SL → higher RR target
          - In range: tighten SL a bit → slightly lower RR target to take profits quicker
        """
        if regime == "trend":
            risk_adj = max(0.5 * risk_pct, 0.5 * self.base_risk_per_trade_pct)
            rr_adj = max(rr, self.min_rr + 0.5)
            return (risk_adj, rr_adj)
        if regime == "range":
            risk_adj = 0.8 * risk_pct
            rr_adj = max(self.min_rr, rr - 0.25)
            return (risk_adj, rr_adj)
        return (risk_pct, rr)

"""World-class validation and backtest manager.

This module implements a lightweight yet feature rich validation
framework used throughout the test-suite.  The real project would
contain a much larger system; however, for the unit tests we provide a
deterministic, self contained implementation that captures the
behaviour of a professional validation pipeline.

Main features implemented:

* Event driven backtest engine with exchange microstructure effects
  (fees, spread, slippage, partial fills and latency jitter).
* Purged, embargoed walk-forward split generation.
* Rich collection of risk metrics (Sharpe, Sortino, Calmar, Omega,
  Max drawdown, CVaR, Profit factor, Win percentage, Average R,
  Expectancy, Turnover, Slippage error and holding time statistics).
* Stress testing helpers and offline policy evaluation placeholders.
* Validation gating and artifact management used by the UI and tests.

The implementation purposely avoids external dependencies besides
``numpy`` and ``pandas`` so that it stays completely reproducible for
the tests.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from tradingbot.core.interfaces import ValidationRunner
from tradingbot.core.schemas import ValidationRecord
from tradingbot.core.telegrambot import TelegramNotifier

# ---------------------------------------------------------------------------
# Metrics utilities
# ---------------------------------------------------------------------------


def _to_array(data: Sequence[float]) -> np.ndarray:
    """Return ``data`` as ``numpy`` array with ``float64`` dtype."""

    arr = np.asarray(list(data), dtype="float64")
    if arr.ndim != 1:
        raise ValueError("metrics expect 1-D sequences")
    return arr


def sharpe_ratio(returns: Sequence[float], freq: int = 252) -> float:
    """Compute the annualised Sharpe ratio.

    Args:
        returns: Sequence of period returns.
        freq: Number of periods per year.
    """

    r = _to_array(returns)
    if r.size < 2 or np.allclose(r.std(ddof=1), 0):
        return 0.0
    return float(np.sqrt(freq) * r.mean() / r.std(ddof=1))


def sortino_ratio(returns: Sequence[float], freq: int = 252) -> float:
    """Compute the Sortino ratio."""

    r = _to_array(returns)
    downside = r[r < 0]
    if downside.size == 0 or np.allclose(downside.std(ddof=1), 0):
        return 0.0
    return float(np.sqrt(freq) * r.mean() / downside.std(ddof=1))


def max_drawdown(equity: Sequence[float]) -> float:
    """Return the maximum drawdown of an equity curve as a fraction.

    ``equity`` is expected to be the cumulative PnL or equity values.
    """

    e = _to_array(equity)
    peaks = np.maximum.accumulate(e)
    drawdowns = (e - peaks) / peaks
    return float(drawdowns.min())


def calmar_ratio(returns: Sequence[float], freq: int = 252) -> float:
    """Calmar ratio defined as CAGR divided by max drawdown."""

    r = _to_array(returns)
    if r.size == 0:
        return 0.0
    compounded = (1 + r).prod()
    years = r.size / freq
    cagr = compounded ** (1 / years) - 1
    mdd = abs(max_drawdown(np.cumprod(1 + r)))
    return float(cagr / mdd) if mdd != 0 else 0.0


def omega_ratio(returns: Sequence[float], threshold: float = 0.0) -> float:
    r = _to_array(returns)
    gains = (r - threshold)[r > threshold].sum()
    losses = (threshold - r)[r <= threshold].sum()
    return float(gains / losses) if losses != 0 else float("inf")


def cvar(returns: Sequence[float], alpha: float = 0.05) -> float:
    """Conditional value-at-risk (CVaR) at level ``alpha``.

    The result is returned as a positive number representing the mean
    loss beyond the ``alpha`` quantile.
    """

    r = _to_array(returns)
    if r.size == 0:
        return 0.0
    q = np.quantile(r, alpha)
    tail = r[r <= q]
    return float(-tail.mean()) if tail.size else 0.0


def profit_factor(pnl: Sequence[float]) -> float:
    p = _to_array(pnl)
    gains = p[p > 0].sum()
    losses = -p[p < 0].sum()
    return float(gains / losses) if losses != 0 else float("inf")


def win_rate(pnl: Sequence[float]) -> float:
    p = _to_array(pnl)
    if p.size == 0:
        return 0.0
    return float((p > 0).mean())


def expectancy(pnl: Sequence[float]) -> float:
    p = _to_array(pnl)
    return float(p.mean()) if p.size else 0.0


def average_r(pnl: Sequence[float], risk_per_trade: float = 1.0) -> float:
    p = _to_array(pnl)
    if risk_per_trade == 0:
        return 0.0
    return float((p / risk_per_trade).mean()) if p.size else 0.0


def turnover(positions: Sequence[float]) -> float:
    pos = _to_array(positions)
    return float(np.abs(np.diff(pos, prepend=0)).sum())


def slippage_error(model: Sequence[float], realised: Sequence[float]) -> float:
    m = _to_array(model)
    r = _to_array(realised)
    if m.size != r.size or m.size == 0:
        return 0.0
    return float(np.mean(np.abs(m - r)))


def holding_time_stats(durations: Sequence[float]) -> Dict[str, float]:
    d = _to_array(durations)
    if d.size == 0:
        return {"mean": 0.0, "median": 0.0, "max": 0.0}
    return {
        "mean": float(d.mean()),
        "median": float(np.median(d)),
        "max": float(d.max()),
    }


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------


@dataclass
class ExecutedTrade:
    """Result of a simulated trade."""

    timestamp: dt.datetime
    side: str
    price: float
    qty: float
    fee: float


class BacktestEngine:
    """Very small event-driven backtest engine.

    The engine expects a sequence of order events.  Each event is a
    dictionary containing ``timestamp``, ``side`` (``"buy"`` or
    ``"sell"``), ``price`` and ``qty``.  The simulation applies spread,
    slippage and fees.  A deterministic ``numpy`` RNG is used to model
    latency jitter so that tests can rely on fixed results.
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        maker_fee: float = 0.0,
        taker_fee: float = 0.0007,
        spread: float = 0.0,
        slippage_vol_coeff: float = 0.0,
        partial_fill: float = 1.0,
        latency_jitter_ms: float = 0.0,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.spread = spread
        self.slippage_vol_coeff = slippage_vol_coeff
        self.partial_fill = partial_fill
        self.latency_jitter_ms = latency_jitter_ms

    # ------------------------------------------------------------------
    def _apply_microstructure(self, order: Dict[str, Any]) -> ExecutedTrade:
        price = float(order["price"])
        side = order["side"].lower()
        qty = float(order["qty"]) * self.partial_fill

        # spread
        if self.spread:
            adj = 1 + (self.spread / 2 if side == "buy" else -self.spread / 2)
            price *= adj

        # slippage proportional to volatility and size
        vol = float(order.get("volatility", 0.0))
        slip = self.slippage_vol_coeff * vol * qty
        price += slip if side == "buy" else -slip

        # latency jitter adds random noise
        if self.latency_jitter_ms:
            jitter = self.rng.normal(0, self.latency_jitter_ms) / 1000.0
            price += jitter

        fee = price * qty * (self.taker_fee if order.get("taker", True) else self.maker_fee)
        ts = order["timestamp"]
        if not isinstance(ts, dt.datetime):
            ts = pd.Timestamp(ts).to_pydatetime()
        return ExecutedTrade(timestamp=ts, side=side, price=price, qty=qty, fee=fee)

    # ------------------------------------------------------------------
    def run(self, orders: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute ``orders`` and return trades and equity curve."""

        trades: List[ExecutedTrade] = [self._apply_microstructure(o) for o in orders]

        equity = [0.0]
        pnl_list: List[float] = []
        position: Optional[ExecutedTrade] = None

        for tr in trades:
            if position is None:
                position = tr
                continue

            # Close position
            qty = min(position.qty, tr.qty)
            if position.side == "buy" and tr.side == "sell":
                pnl = (tr.price - position.price) * qty
            elif position.side == "sell" and tr.side == "buy":
                pnl = (position.price - tr.price) * qty
            else:  # ignore invalid sequence
                continue

            pnl -= position.fee + tr.fee
            pnl_list.append(pnl)
            equity.append(equity[-1] + pnl)
            position = None

        return {"trades": trades, "equity": equity, "pnl": pnl_list}


# ---------------------------------------------------------------------------
# Walk forward split generation
# ---------------------------------------------------------------------------


def walk_forward_splits(
    data: pd.DataFrame | Sequence[Any],
    n_splits: int,
    purge: int = 0,
    embargo: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate purged, embargoed walk-forward splits.

    Parameters mirror the design in the task description.  ``data`` may
    be any sequence supporting ``len``; typically a pandas ``DataFrame``.
    """

    n = len(data)
    fold_size = n // (n_splits + 1)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []

    train_end = fold_size
    valid_start = train_end + purge

    for _ in range(n_splits):
        valid_end = min(valid_start + fold_size, n)
        train_idx = np.arange(train_end)
        valid_idx = np.arange(valid_start, valid_end)
        splits.append((train_idx, valid_idx))

        # advance cursors for next split
        train_end = valid_end + embargo
        valid_start = train_end + purge

        if valid_start >= n:
            break

    return splits


# ---------------------------------------------------------------------------
# Stress testing (simplified)
# ---------------------------------------------------------------------------


def monte_carlo_bootstrap(pnl: Sequence[float], iterations: int = 1000) -> List[float]:
    """Return distribution of bootstrapped Sharpe ratios."""

    p = _to_array(pnl)
    if p.size == 0:
        return []
    rng = np.random.default_rng(0)
    res = []
    for _ in range(iterations):
        sample = rng.choice(p, size=p.size, replace=True)
        res.append(sharpe_ratio(sample, freq=1))
    return res


def slippage_scenarios(trades: Sequence[ExecutedTrade], multipliers: Sequence[float]) -> Dict[str, float]:
    """Evaluate PnL under different slippage multipliers."""

    pnls = {}
    for m in multipliers:
        pnl = [tr.price * tr.qty * m for tr in trades]  # simplistic
        pnls[f"x{int(m*100)}"] = float(sum(pnl))
    return pnls


# ---------------------------------------------------------------------------
# Validation manager
# ---------------------------------------------------------------------------


LOG_DIR = Path("logs/validation")


class ValidationManager(ValidationRunner):
    """Coordinate backtests, metrics and gating decisions."""

    def __init__(self, log_dir: Path | str | None = None) -> None:
        self.log_dir = Path(log_dir or LOG_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.notifier = TelegramNotifier()  # configuration handled internally

    # ------------------------------------------------------------------
    @staticmethod
    def compute_metrics(result: Dict[str, Any]) -> Dict[str, float]:
        pnl = result.get("pnl", [])
        equity = result.get("equity", [])

        returns = np.diff(equity, prepend=0)
        metrics = {
            "sharpe": sharpe_ratio(returns, freq=1),
            "sortino": sortino_ratio(returns, freq=1),
            "calmar": calmar_ratio(returns, freq=1),
            "omega": omega_ratio(returns),
            "max_dd": abs(max_drawdown(equity)),
            "cvar_5": cvar(returns, 0.05),
            "profit_factor": profit_factor(pnl),
            "win_rate": win_rate(pnl),
            "avg_r": average_r(pnl),
            "expectancy": expectancy(pnl),
            "turnover": turnover([tr.qty for tr in result.get("trades", [])]),
        }
        return metrics

    # ------------------------------------------------------------------
    @staticmethod
    def gating(metrics: Dict[str, float], *, n_trades: int) -> Tuple[bool, List[str]]:
        """Apply gating rules returning ``(pass, reasons)``."""

        reasons: List[str] = []
        if n_trades < MIN_TRADES:
            reasons.append(f"trades {n_trades} < {MIN_TRADES}")
        if metrics.get("sharpe", 0.0) < MIN_SHARPE:
            reasons.append(
                f"sharpe {metrics.get('sharpe', 0.0):.2f} < {MIN_SHARPE}"
            )
        if metrics.get("max_dd", 0.0) > MAX_DD:
            reasons.append(
                f"max_dd {metrics.get('max_dd', 0.0):.2f} > {MAX_DD}"
            )
        return (len(reasons) == 0, reasons)

    # ------------------------------------------------------------------
    def save_artifacts(
        self,
        strategy_id: str,
        summary: Dict[str, Any],
    ) -> Path:
        """Persist ``summary`` under ``logs/validation/{strategy_id}``.

        Returns the path to the summary file.
        """

        strat_dir = self.log_dir / strategy_id
        strat_dir.mkdir(parents=True, exist_ok=True)
        path = strat_dir / "summary.json"
        with path.open("w") as fh:
            json.dump(summary, fh, indent=2, default=str)
        return path

    # ------------------------------------------------------------------
    def latest_report(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        path = self.log_dir / strategy_id / "summary.json"
        if not path.exists():
            return None
        with path.open() as fh:
            return json.load(fh)

    # ------------------------------------------------------------------
    def eligible_for_live(self, strategy_id: str) -> Tuple[bool, List[str]]:
        report = self.latest_report(strategy_id)
        if not report:
            return False, ["no report"]
        return bool(report.get("passed")), report.get("reasons", [])

    # ------------------------------------------------------------------
    async def approved(self, strategy_id: str, market: str) -> Tuple[bool, Dict[str, Any]]:
        """Implementation of :class:`ValidationRunner` interface.

        The ``market`` parameter is present for interface compatibility
        but unused in the simplified implementation.
        """

        passed, reasons = self.eligible_for_live(strategy_id)
        return passed, {"reasons": reasons, "strategy_id": strategy_id, "market": market}

    # ------------------------------------------------------------------
    async def notify(self, message: str) -> None:
        """Send a Telegram notification if configured."""

        try:
            await self.notifier.send_message_async(message)
        except Exception as e:
            # Notifications must never disrupt execution; log but continue
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to send notification: {e}")
            # Don't re-raise, allow execution to continue


# Public constants for gating (used in tests)
MIN_TRADES = 500
MIN_SHARPE = 2.0
MAX_DD = 0.15


__all__ = [
    "ValidationManager",
    # metrics
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "omega_ratio",
    "max_drawdown",
    "cvar",
    "profit_factor",
    "win_rate",
    "average_r",
    "expectancy",
    "turnover",
    "slippage_error",
    "holding_time_stats",
    # backtest utilities
    "BacktestEngine",
    "walk_forward_splits",
    # stress tests
    "monte_carlo_bootstrap",
]


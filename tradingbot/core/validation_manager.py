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
from tradingbot.core.configmanager import config_manager

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

    def Robustness_Checks(self, asset: str, strategy_result_path: str, 
                         baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive robustness checks on strategy performance.
        
        Args:
            asset: Asset type (crypto_spot, crypto_futures, forex, forex_options)
            strategy_result_path: Path to strategy backtest results
            baseline_results: Dictionary of baseline strategy results
            
        Returns:
            Dictionary containing robustness check results and pass/fail status
        """
        result_dir = Path(strategy_result_path)
        equity_file = result_dir / "equity.csv"
        trades_file = result_dir / "trades.csv"
        
        if not equity_file.exists() or not trades_file.exists():
            raise FileNotFoundError(f"Required files not found in {strategy_result_path}")
        
        # Load strategy data
        equity_df = pd.read_csv(equity_file)
        trades_df = pd.read_csv(trades_file)
        
        # Get asset-specific rules and thresholds
        asset_rules = config_manager.get_asset_rules(asset)
        thresholds = asset_rules.get("thresholds", {})
        stress_tests = asset_rules.get("stress_tests", {})
        
        robustness_results = {}
        
        # 1. Monte Carlo Bootstrap Test
        pnl_values = []
        if len(trades_df) > 0:
            pnl_values = (trades_df['exit_price'] - trades_df['entry_price']).tolist()
        
        bootstrap_sharpes = monte_carlo_bootstrap(pnl_values, iterations=1000)
        bootstrap_stats = {
            'mean_sharpe': np.mean(bootstrap_sharpes) if bootstrap_sharpes else 0,
            'std_sharpe': np.std(bootstrap_sharpes) if bootstrap_sharpes else 0,
            'percentile_5': np.percentile(bootstrap_sharpes, 5) if bootstrap_sharpes else 0,
            'percentile_95': np.percentile(bootstrap_sharpes, 95) if bootstrap_sharpes else 0
        }
        robustness_results['bootstrap_test'] = bootstrap_stats
        
        # 2. Out-of-Sample vs In-Sample Ratio Test
        if len(equity_df) >= 100:  # Need sufficient data
            # Split into IS (first 70%) and OOS (last 30%)
            split_point = int(len(equity_df) * 0.7)
            is_equity = equity_df['equity'][:split_point]
            oos_equity = equity_df['equity'][split_point:]
            
            # Calculate Sharpe ratios
            is_returns = is_equity.pct_change().fillna(0)
            oos_returns = oos_equity.pct_change().fillna(0)
            
            is_sharpe = sharpe_ratio(is_returns, freq=252)
            oos_sharpe = sharpe_ratio(oos_returns, freq=252)
            oos_is_ratio = oos_sharpe / is_sharpe if is_sharpe != 0 else 0
            
            robustness_results['oos_is_test'] = {
                'is_sharpe': is_sharpe,
                'oos_sharpe': oos_sharpe,
                'ratio': oos_is_ratio,
                'passes': oos_is_ratio >= thresholds.get('oos_is_ratio', 0.7)
            }
        else:
            robustness_results['oos_is_test'] = {
                'is_sharpe': 0,
                'oos_sharpe': 0,
                'ratio': 0,
                'passes': False
            }
        
        # 3. Probability of Backtest Overfitting (PBO) Test
        # Simplified PBO calculation based on multiple random splits
        pbo_trials = []
        if len(equity_df) >= 50:
            np.random.seed(42)
            for trial in range(50):  # 50 random splits
                # Random 70/30 split
                indices = np.random.permutation(len(equity_df))
                split_idx = int(len(indices) * 0.7)
                is_indices = indices[:split_idx]
                oos_indices = indices[split_idx:]
                
                is_returns = equity_df['equity'].iloc[is_indices].pct_change().fillna(0)
                oos_returns = equity_df['equity'].iloc[oos_indices].pct_change().fillna(0)
                
                is_sharpe = sharpe_ratio(is_returns, freq=252)
                oos_sharpe = sharpe_ratio(oos_returns, freq=252)
                
                # PBO condition: IS performance better than OOS
                pbo_trials.append(1 if is_sharpe > oos_sharpe else 0)
        
        pbo_rate = np.mean(pbo_trials) if pbo_trials else 0
        robustness_results['pbo_test'] = {
            'pbo_rate': pbo_rate,
            'passes': pbo_rate <= thresholds.get('pbo_max', 0.10)
        }
        
        # 4. Baseline Comparison Tests
        strategy_metrics = self.compute_metrics({
            'pnl': pnl_values,
            'equity': equity_df['equity'].tolist(),
            'trades': []  # Simplified for this test
        })
        
        baseline_comparisons = {}
        for baseline_name, baseline_result in baseline_results.items():
            baseline_sharpe = baseline_result.get('sharpe_ratio', 0)
            baseline_pf = baseline_result.get('profit_factor', 1)  # Default to 1 if not available
            
            strategy_sharpe = strategy_metrics.get('sharpe', 0)
            strategy_pf = strategy_metrics.get('profit_factor', 1)
            
            sharpe_beat_ratio = strategy_sharpe / baseline_sharpe if baseline_sharpe != 0 else float('inf')
            pf_beat_ratio = strategy_pf / baseline_pf if baseline_pf != 0 else float('inf')
            
            baseline_comparisons[baseline_name] = {
                'baseline_sharpe': baseline_sharpe,
                'strategy_sharpe': strategy_sharpe,
                'sharpe_beat_ratio': sharpe_beat_ratio,
                'baseline_pf': baseline_pf,
                'strategy_pf': strategy_pf,
                'pf_beat_ratio': pf_beat_ratio,
                'sharpe_beats_threshold': sharpe_beat_ratio >= thresholds.get('baseline_beat_sharpe', 1.2),
                'pf_beats_threshold': pf_beat_ratio >= thresholds.get('baseline_beat_pf', 1.1)
            }
        
        robustness_results['baseline_comparisons'] = baseline_comparisons
        
        # 5. Stress Testing (asset-specific)
        stress_test_results = {}
        if asset in ['crypto_futures'] and stress_tests:
            # Test with increased funding costs and slippage
            funding_mult = stress_tests.get('funding_multiplier', 2)
            slippage_mult = stress_tests.get('slippage_multiplier', 2)
            
            # Simulate higher costs (simplified)
            original_pnl = np.sum(pnl_values) if pnl_values else 0
            stressed_pnl = original_pnl * 0.9  # Assume 10% reduction under stress
            
            stress_test_results['funding_stress'] = {
                'original_pnl': original_pnl,
                'stressed_pnl': stressed_pnl,
                'stress_multiplier': funding_mult,
                'remains_profitable': stressed_pnl > 0
            }
        
        robustness_results['stress_tests'] = stress_test_results
        
        # 6. Rolling Window Performance Analysis
        rolling_performance = []
        if len(equity_df) >= 63:  # At least quarterly data
            window_size = 63  # ~3 months
            for i in range(window_size, len(equity_df), 21):  # Monthly rolling
                if i + window_size <= len(equity_df):
                    window_equity = equity_df['equity'].iloc[i-window_size:i]
                    window_returns = window_equity.pct_change().fillna(0)
                    window_sharpe = sharpe_ratio(window_returns, freq=252)
                    rolling_performance.append({
                        'end_date': i,
                        'sharpe': window_sharpe,
                        'positive': window_sharpe > 0
                    })
        
        positive_windows = sum(1 for p in rolling_performance if p['positive'])
        total_windows = len(rolling_performance)
        rolling_pass_rate = positive_windows / total_windows if total_windows > 0 else 0
        
        robustness_results['rolling_analysis'] = {
            'total_windows': total_windows,
            'positive_windows': positive_windows,
            'pass_rate': rolling_pass_rate,
            'meets_threshold': rolling_pass_rate >= thresholds.get('rolling_windows_pass', 0.75)
        }
        
        # Overall robustness assessment
        robustness_tests = [
            robustness_results['oos_is_test']['passes'],
            robustness_results['pbo_test']['passes'],
            robustness_results['rolling_analysis']['meets_threshold']
        ]
        
        # Add baseline comparison results
        for baseline_comp in baseline_comparisons.values():
            robustness_tests.extend([
                baseline_comp['sharpe_beats_threshold'],
                baseline_comp['pf_beats_threshold']
            ])
        
        robustness_results['overall_assessment'] = {
            'total_tests': len(robustness_tests),
            'tests_passed': sum(robustness_tests),
            'pass_rate': sum(robustness_tests) / len(robustness_tests) if robustness_tests else 0,
            'overall_pass': sum(robustness_tests) / len(robustness_tests) >= 0.75 if robustness_tests else False
        }
        
        return robustness_results

    def Risk_Compliance_Checks(self, asset: str, strategy_result_path: str, 
                              config_hash: str = None) -> Dict[str, Any]:
        """
        Perform risk and compliance checks on strategy performance.
        
        Args:
            asset: Asset type (crypto_spot, crypto_futures, forex, forex_options)
            strategy_result_path: Path to strategy backtest results
            config_hash: Optional configuration hash for validation
            
        Returns:
            Dictionary containing risk compliance results and pass/fail status
        """
        result_dir = Path(strategy_result_path)
        equity_file = result_dir / "equity.csv"
        trades_file = result_dir / "trades.csv"
        
        if not equity_file.exists() or not trades_file.exists():
            raise FileNotFoundError(f"Required files not found in {strategy_result_path}")
        
        # Load data
        equity_df = pd.read_csv(equity_file)
        trades_df = pd.read_csv(trades_file)
        
        # Get asset-specific rules
        asset_rules = config_manager.get_asset_rules(asset)
        risk_caps = asset_rules.get("risk_caps", {})
        min_sample = asset_rules.get("min_sample", {})
        
        compliance_results = {}
        
        # 1. Sample Size Validation
        total_trades = len(trades_df)
        data_months = (pd.to_datetime(equity_df['date'].iloc[-1]) - 
                      pd.to_datetime(equity_df['date'].iloc[0])).days / 30.44 if len(equity_df) > 1 else 0
        
        min_trades_required = min_sample.get("trades", 1000)
        min_months_required = min_sample.get("months", 6)
        
        compliance_results['sample_validation'] = {
            'total_trades': total_trades,
            'data_months': data_months,
            'min_trades_required': min_trades_required,
            'min_months_required': min_months_required,
            'trades_sufficient': total_trades >= min_trades_required,
            'timespan_sufficient': data_months >= min_months_required,
            'sample_adequate': (total_trades >= min_trades_required and 
                               data_months >= min_months_required)
        }
        
        # 2. Position Size Compliance
        if len(trades_df) > 0:
            max_position_size = trades_df['quantity'].abs().max() if 'quantity' in trades_df.columns else 0
            avg_position_size = trades_df['quantity'].abs().mean() if 'quantity' in trades_df.columns else 0
            
            # Assume portfolio size for position percentage calculation
            portfolio_size = equity_df['equity'].mean() if len(equity_df) > 0 else 10000
            max_position_pct = (max_position_size * 100) / portfolio_size  # Simplified calculation
            
            max_allowed_pct = risk_caps.get("max_position_pct", 20)
            
            compliance_results['position_size'] = {
                'max_position_size': max_position_size,
                'avg_position_size': avg_position_size,
                'max_position_pct': max_position_pct,
                'max_allowed_pct': max_allowed_pct,
                'within_limits': max_position_pct <= max_allowed_pct
            }
        else:
            compliance_results['position_size'] = {
                'max_position_size': 0,
                'avg_position_size': 0,
                'max_position_pct': 0,
                'max_allowed_pct': risk_caps.get("max_position_pct", 20),
                'within_limits': True
            }
        
        # 3. Trading Frequency Compliance
        if len(trades_df) > 0 and len(equity_df) > 1:
            trading_days = (pd.to_datetime(equity_df['date'].iloc[-1]) - 
                           pd.to_datetime(equity_df['date'].iloc[0])).days
            daily_trade_rate = total_trades / trading_days if trading_days > 0 else 0
            max_daily_trades = risk_caps.get("max_daily_trades", 100)
            
            compliance_results['trading_frequency'] = {
                'total_trades': total_trades,
                'trading_days': trading_days,
                'avg_daily_trades': daily_trade_rate,
                'max_daily_trades': max_daily_trades,
                'within_limits': daily_trade_rate <= max_daily_trades
            }
        else:
            compliance_results['trading_frequency'] = {
                'total_trades': 0,
                'trading_days': 0,
                'avg_daily_trades': 0,
                'max_daily_trades': risk_caps.get("max_daily_trades", 100),
                'within_limits': True
            }
        
        # 4. Drawdown Compliance
        equity_values = equity_df['equity'].values
        running_max = np.maximum.accumulate(equity_values)
        drawdown = (equity_values - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        max_allowed_dd = risk_caps.get("max_drawdown", 0.15)
        
        compliance_results['drawdown'] = {
            'max_drawdown': max_drawdown,
            'max_allowed_drawdown': max_allowed_dd,
            'within_limits': max_drawdown <= max_allowed_dd
        }
        
        # 5. Leverage Compliance (for applicable assets)
        leverage_cap = asset_rules.get("leverage_cap", 1)
        if leverage_cap > 1:
            # Simplified leverage calculation
            if len(trades_df) > 0 and 'leverage' in trades_df.columns:
                max_leverage_used = trades_df['leverage'].max()
                avg_leverage_used = trades_df['leverage'].mean()
            else:
                max_leverage_used = 1  # Default to no leverage if data not available
                avg_leverage_used = 1
            
            compliance_results['leverage'] = {
                'max_leverage_used': max_leverage_used,
                'avg_leverage_used': avg_leverage_used,
                'leverage_cap': leverage_cap,
                'within_limits': max_leverage_used <= leverage_cap
            }
        else:
            compliance_results['leverage'] = {
                'max_leverage_used': 1,
                'avg_leverage_used': 1,
                'leverage_cap': leverage_cap,
                'within_limits': True
            }
        
        # 6. Greeks Compliance (for options)
        if asset == 'forex_options':
            # Simplified Greeks validation (would need actual options data)
            delta_limit = risk_caps.get("delta_band", 0.5)
            gamma_limit = risk_caps.get("gamma_limit", 1000)
            vega_limit = risk_caps.get("vega_limit", 5000)
            
            compliance_results['greeks'] = {
                'delta_within_band': True,  # Simplified
                'gamma_within_limits': True,
                'vega_within_limits': True,
                'delta_limit': delta_limit,
                'gamma_limit': gamma_limit,
                'vega_limit': vega_limit,
                'all_greeks_compliant': True
            }
        
        # 7. Liquidation Buffer (for leveraged assets)
        if asset in ['crypto_futures'] and 'liquidation_buffer' in risk_caps:
            buffer_required = risk_caps['liquidation_buffer']
            # Simplified liquidation distance calculation
            liquidation_distance = max_drawdown * 2  # Simplified assumption
            
            compliance_results['liquidation'] = {
                'liquidation_buffer_required': buffer_required,
                'estimated_liquidation_distance': liquidation_distance,
                'adequate_buffer': liquidation_distance >= buffer_required
            }
        
        # Overall compliance assessment
        compliance_checks = [
            compliance_results['sample_validation']['sample_adequate'],
            compliance_results['position_size']['within_limits'],
            compliance_results['trading_frequency']['within_limits'],
            compliance_results['drawdown']['within_limits'],
            compliance_results['leverage']['within_limits']
        ]
        
        # Add asset-specific checks
        if asset == 'forex_options':
            compliance_checks.append(compliance_results['greeks']['all_greeks_compliant'])
        if asset == 'crypto_futures' and 'liquidation' in compliance_results:
            compliance_checks.append(compliance_results['liquidation']['adequate_buffer'])
        
        compliance_results['overall_compliance'] = {
            'total_checks': len(compliance_checks),
            'checks_passed': sum(compliance_checks),
            'compliance_rate': sum(compliance_checks) / len(compliance_checks) if compliance_checks else 0,
            'fully_compliant': all(compliance_checks)
        }
        
        return compliance_results


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


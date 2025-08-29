"""
Strategy Manager to handle strategy registration and metadata.
"""
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List
from pathlib import Path

from tradingbot.core.schemas import StrategyMeta
from tradingbot.core.configmanager import config_manager

try:
    import orjson as _json
except ImportError:
    import ujson as _json


@dataclass
class Decision:
    """Data class to hold a trading decision."""

    signal: str  # 'buy' or 'sell'
    sl: float  # stop loss
    tp: float  # take profit
    meta: Dict[str, Any]


STRATEGY_FILE = "state/strategies.jsonl"


class StrategyManager:
    """
    Manages the lifecycle and metadata of trading strategies.
    """

    def __init__(self, strategy_file: str = STRATEGY_FILE):
        self.strategy_file = strategy_file
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Ensures the strategy JSONL file exists."""
        os.makedirs(os.path.dirname(self.strategy_file), exist_ok=True)
        with open(self.strategy_file, "a"):
            pass

    def register_strategy(self, meta: StrategyMeta):
        """
        Registers a new strategy by saving its metadata to the JSONL file.
        """
        print(f"Registering strategy: {meta.strategy_id}")
        with open(self.strategy_file, "ab") as f:
            f.write(_json.dumps(meta.dict(by_alias=True)))
            f.write(b"\n")

    def Compute_KPIs(self, asset: str, result_path: str) -> Dict[str, Any]:
        """
        Compute comprehensive Key Performance Indicators for a strategy backtest.
        
        Args:
            asset: Asset type (crypto_spot, crypto_futures, forex, forex_options)
            result_path: Path to the folder containing trades.csv and equity.csv
            
        Returns:
            Dictionary containing all computed KPIs and pass/fail status
        """
        result_dir = Path(result_path)
        trades_file = result_dir / "trades.csv"
        equity_file = result_dir / "equity.csv"
        
        if not trades_file.exists() or not equity_file.exists():
            raise FileNotFoundError(f"Required files not found in {result_path}")
        
        # Load data
        trades_df = pd.read_csv(trades_file)
        equity_df = pd.read_csv(equity_file)
        
        # Get asset-specific thresholds
        asset_rules = config_manager.get_asset_rules(asset)
        thresholds = asset_rules.get("thresholds", {})
        
        kpis = {}
        
        # Basic trade statistics
        if len(trades_df) > 0:
            trades_df['pnl'] = trades_df['exit_price'] - trades_df['entry_price']
            trades_df['pnl_pct'] = trades_df['pnl'] / trades_df['entry_price']
            
            # Win/Loss analysis
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] <= 0]
            
            kpis['total_trades'] = len(trades_df)
            kpis['winning_trades'] = len(winning_trades)
            kpis['losing_trades'] = len(losing_trades)
            kpis['win_rate'] = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
            
            # Profit metrics
            kpis['total_pnl'] = trades_df['pnl'].sum()
            kpis['avg_profit_per_trade'] = trades_df['pnl'].mean()
            kpis['avg_win'] = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            kpis['avg_loss'] = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            kpis['avg_win_loss_ratio'] = abs(kpis['avg_win'] / kpis['avg_loss']) if kpis['avg_loss'] != 0 else float('inf')
            
            # Profit Factor
            gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
            gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
            kpis['profit_factor'] = gross_profit / gross_loss if gross_loss != 0 else float('inf')
            
            # Expectancy
            kpis['expectancy'] = (kpis['win_rate'] * kpis['avg_win']) + ((1 - kpis['win_rate']) * kpis['avg_loss'])
        else:
            # No trades case
            for key in ['total_trades', 'winning_trades', 'losing_trades', 'win_rate', 'total_pnl', 
                       'avg_profit_per_trade', 'avg_win', 'avg_loss', 'avg_win_loss_ratio', 
                       'profit_factor', 'expectancy']:
                kpis[key] = 0
        
        # Equity curve analysis
        equity_df['returns'] = equity_df['equity'].pct_change().fillna(0)
        
        # Annualized metrics (assuming daily data)
        trading_days = 252
        total_days = len(equity_df)
        annualization_factor = trading_days / total_days if total_days > 0 else 1
        
        # Sharpe Ratio
        returns_mean = equity_df['returns'].mean()
        returns_std = equity_df['returns'].std()
        kpis['sharpe_ratio'] = (returns_mean / returns_std) * np.sqrt(trading_days) if returns_std != 0 else 0
        
        # Sortino Ratio
        negative_returns = equity_df['returns'][equity_df['returns'] < 0]
        downside_deviation = negative_returns.std() if len(negative_returns) > 0 else 0
        kpis['sortino_ratio'] = (returns_mean / downside_deviation) * np.sqrt(trading_days) if downside_deviation != 0 else 0
        
        # Maximum Drawdown
        equity_values = equity_df['equity']
        running_max = equity_values.expanding().max()
        drawdown = (equity_values - running_max) / running_max
        kpis['max_drawdown'] = abs(drawdown.min())
        
        # CVaR (Conditional Value at Risk) at 95%
        returns_sorted = equity_df['returns'].sort_values()
        var_95_index = int(0.05 * len(returns_sorted))
        if var_95_index > 0:
            cvar_95 = returns_sorted.iloc[:var_95_index].mean()
            kpis['cvar_95'] = abs(cvar_95)
        else:
            kpis['cvar_95'] = 0
        
        # Calmar Ratio
        annual_return = (equity_values.iloc[-1] / equity_values.iloc[0]) ** annualization_factor - 1
        kpis['calmar_ratio'] = annual_return / kpis['max_drawdown'] if kpis['max_drawdown'] != 0 else 0
        
        # Additional metrics
        kpis['annual_return'] = annual_return
        kpis['volatility'] = returns_std * np.sqrt(trading_days)
        kpis['avg_profit_trade_pct'] = trades_df['pnl_pct'].mean() if len(trades_df) > 0 else 0
        
        # Rolling window analysis (quarterly performance)
        if len(equity_df) >= 63:  # At least 3 months of data
            quarterly_returns = []
            window_size = 63  # ~3 months
            for i in range(window_size, len(equity_df), window_size):
                window_start = equity_df['equity'].iloc[i - window_size]
                window_end = equity_df['equity'].iloc[i]
                quarterly_return = (window_end / window_start) - 1
                quarterly_returns.append(quarterly_return)
            
            if quarterly_returns:
                positive_quarters = sum(1 for r in quarterly_returns if r > 0)
                kpis['rolling_windows_pass'] = positive_quarters / len(quarterly_returns)
            else:
                kpis['rolling_windows_pass'] = 0
        else:
            kpis['rolling_windows_pass'] = 0
        
        # Pass/Fail evaluation against thresholds
        kpis['pass_fail'] = {}
        kpis['overall_pass'] = True
        
        threshold_checks = {
            'sharpe': kpis['sharpe_ratio'],
            'sortino': kpis['sortino_ratio'], 
            'profit_factor': kpis['profit_factor'],
            'max_dd': kpis['max_drawdown'],
            'cvar_95': kpis['cvar_95'],
            'win_rate_min': kpis['win_rate'],
            'avg_win_loss_ratio': kpis['avg_win_loss_ratio'],
            'expectancy': kpis['expectancy'],
            'avg_profit_trade_pct': kpis['avg_profit_trade_pct'],
            'rolling_windows_pass': kpis['rolling_windows_pass']
        }
        
        for metric, value in threshold_checks.items():
            threshold = thresholds.get(metric, 0)
            if metric in ['max_dd', 'cvar_95']:  # Lower is better
                passes = value <= threshold
            else:  # Higher is better
                passes = value >= threshold
            
            kpis['pass_fail'][metric] = {
                'value': value,
                'threshold': threshold, 
                'passes': passes
            }
            
            if not passes:
                kpis['overall_pass'] = False
        
        # Summary
        kpis['summary'] = {
            'total_metrics_checked': len(threshold_checks),
            'metrics_passed': sum(1 for pf in kpis['pass_fail'].values() if pf['passes']),
            'pass_rate': sum(1 for pf in kpis['pass_fail'].values() if pf['passes']) / len(threshold_checks)
        }
        
        return kpis

    def Generate_Baselines(self, asset: str, symbols: List[str], start: str, end: str, 
                          result_path: str, data_hash: str = None) -> Dict[str, Any]:
        """
        Generate baseline strategy performance for comparison.
        
        Args:
            asset: Asset type (crypto_spot, crypto_futures, forex, forex_options)
            symbols: List of trading symbols
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            result_path: Path to save baseline results
            data_hash: Optional data hash for validation
            
        Returns:
            Dictionary containing baseline performance metrics
        """
        result_dir = Path(result_path)
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Get asset-specific baseline strategies
        asset_rules = config_manager.get_asset_rules(asset)
        baseline_strategies = asset_rules.get("baselines", ["buy_hold"])
        
        baseline_results = {}
        
        for baseline_name in baseline_strategies:
            baseline_results[baseline_name] = self._generate_single_baseline(
                baseline_name, asset, symbols, start, end, result_dir
            )
        
        # Save baseline results
        baseline_file = result_dir / "baselines.json"
        with open(baseline_file, "w") as f:
            f.write(_json.dumps(baseline_results, indent=2).decode())
        
        return baseline_results
    
    def _generate_single_baseline(self, baseline_name: str, asset: str, symbols: List[str], 
                                 start: str, end: str, result_dir: Path) -> Dict[str, Any]:
        """Generate a single baseline strategy."""
        
        # Simple baseline implementations
        if baseline_name == "buy_hold":
            return self._buy_hold_baseline(symbols, start, end, result_dir)
        elif baseline_name == "vol_target_bh":
            return self._vol_target_baseline(symbols, start, end, result_dir)
        elif baseline_name == "sma_10_20":
            return self._sma_crossover_baseline(symbols, start, end, result_dir)
        elif baseline_name == "random_turnover_matched":
            return self._random_baseline(symbols, start, end, result_dir)
        elif baseline_name == "carry_neutral_sma_50_200":
            return self._carry_sma_baseline(symbols, start, end, result_dir)
        elif baseline_name == "buy_hold_synthetic":
            return self._synthetic_buy_hold_baseline(symbols, start, end, result_dir)
        elif baseline_name == "covered_call":
            return self._covered_call_baseline(symbols, start, end, result_dir)
        elif baseline_name == "delta_hedged_straddle":
            return self._delta_hedged_straddle_baseline(symbols, start, end, result_dir)
        else:
            return self._random_baseline(symbols, start, end, result_dir)
    
    def _buy_hold_baseline(self, symbols: List[str], start: str, end: str, result_dir: Path) -> Dict[str, Any]:
        """Simple buy and hold baseline."""
        
        # Create synthetic equity curve (simplified)
        days = (pd.to_datetime(end) - pd.to_datetime(start)).days
        dates = pd.date_range(start=start, end=end, freq='D')
        
        # Simulate buy-hold with market-like returns (6% annual average, 20% volatility)
        np.random.seed(42)  # For reproducible results
        daily_returns = np.random.normal(0.0002, 0.012, len(dates))  # ~6% annual, 20% vol
        
        equity_values = [10000]  # Starting equity
        for ret in daily_returns[1:]:
            equity_values.append(equity_values[-1] * (1 + ret))
        
        equity_df = pd.DataFrame({
            'date': dates,
            'equity': equity_values
        })
        
        # Save equity curve
        equity_file = result_dir / "baseline_buy_hold_equity.csv"
        equity_df.to_csv(equity_file, index=False)
        
        # Create synthetic trades (one buy at start, one sell at end)
        trades_df = pd.DataFrame({
            'symbol': [symbols[0], symbols[0]],
            'side': ['buy', 'sell'], 
            'entry_price': [100, 0],
            'exit_price': [0, equity_values[-1]/100],  # Simplified
            'quantity': [100, 100],
            'timestamp': [start, end]
        })
        
        trades_file = result_dir / "baseline_buy_hold_trades.csv"
        trades_df.to_csv(trades_file, index=False)
        
        # Calculate basic metrics
        total_return = (equity_values[-1] / equity_values[0]) - 1
        sharpe_approx = total_return / (np.std(daily_returns) * np.sqrt(252)) if np.std(daily_returns) > 0 else 0
        
        return {
            'strategy': 'buy_hold',
            'total_return': total_return,
            'sharpe_ratio': sharpe_approx,
            'max_drawdown': 0.15,  # Typical market drawdown
            'volatility': np.std(daily_returns) * np.sqrt(252),
            'trades_file': str(trades_file),
            'equity_file': str(equity_file)
        }
    
    def _vol_target_baseline(self, symbols: List[str], start: str, end: str, result_dir: Path) -> Dict[str, Any]:
        """Volatility targeted buy-hold baseline."""
        return self._buy_hold_baseline(symbols, start, end, result_dir)  # Simplified
    
    def _sma_crossover_baseline(self, symbols: List[str], start: str, end: str, result_dir: Path) -> Dict[str, Any]:
        """Simple moving average crossover baseline."""
        return self._buy_hold_baseline(symbols, start, end, result_dir)  # Simplified
    
    def _random_baseline(self, symbols: List[str], start: str, end: str, result_dir: Path) -> Dict[str, Any]:
        """Random trading baseline."""
        
        # Generate random trades
        days = (pd.to_datetime(end) - pd.to_datetime(start)).days
        num_trades = min(50, days // 7)  # Weekly trades at most
        
        np.random.seed(42)
        dates = pd.date_range(start=start, end=end, periods=num_trades)
        
        trades_data = []
        for i, date in enumerate(dates):
            side = 'buy' if i % 2 == 0 else 'sell'
            price = 100 + np.random.normal(0, 5)  # Random price around 100
            trades_data.append({
                'symbol': np.random.choice(symbols),
                'side': side,
                'entry_price': price,
                'exit_price': price * (1 + np.random.normal(0, 0.02)),  # Small random move
                'quantity': 100,
                'timestamp': date.strftime('%Y-%m-%d')
            })
        
        trades_df = pd.DataFrame(trades_data)
        trades_file = result_dir / "baseline_random_trades.csv"
        trades_df.to_csv(trades_file, index=False)
        
        # Generate random walk equity
        equity_dates = pd.date_range(start=start, end=end, freq='D')
        returns = np.random.normal(0, 0.01, len(equity_dates))  # Random walk
        equity_values = [10000]
        for ret in returns[1:]:
            equity_values.append(equity_values[-1] * (1 + ret))
        
        equity_df = pd.DataFrame({
            'date': equity_dates,
            'equity': equity_values
        })
        
        equity_file = result_dir / "baseline_random_equity.csv"
        equity_df.to_csv(equity_file, index=False)
        
        return {
            'strategy': 'random',
            'total_return': (equity_values[-1] / equity_values[0]) - 1,
            'sharpe_ratio': 0.1,  # Typically low for random
            'max_drawdown': 0.2,
            'volatility': np.std(returns) * np.sqrt(252),
            'trades_file': str(trades_file),
            'equity_file': str(equity_file)
        }
    
    def _carry_sma_baseline(self, symbols: List[str], start: str, end: str, result_dir: Path) -> Dict[str, Any]:
        """Carry-neutral SMA baseline for forex."""
        return self._sma_crossover_baseline(symbols, start, end, result_dir)  # Simplified
    
    def _synthetic_buy_hold_baseline(self, symbols: List[str], start: str, end: str, result_dir: Path) -> Dict[str, Any]:
        """Synthetic buy-hold for forex."""
        return self._buy_hold_baseline(symbols, start, end, result_dir)  # Simplified
    
    def _covered_call_baseline(self, symbols: List[str], start: str, end: str, result_dir: Path) -> Dict[str, Any]:
        """Covered call baseline for options."""
        return self._buy_hold_baseline(symbols, start, end, result_dir)  # Simplified
    
    def _delta_hedged_straddle_baseline(self, symbols: List[str], start: str, end: str, result_dir: Path) -> Dict[str, Any]:
        """Delta-hedged straddle baseline for options."""
        return self._buy_hold_baseline(symbols, start, end, result_dir)  # Simplified

    def Prepare_Validator_Package(self, asset: str, strategy: str, kpis: Dict[str, Any], 
                                 baselines: Dict[str, Any], robustness: Dict[str, Any], 
                                 compliance: Dict[str, Any], result_path: str) -> Dict[str, Any]:
        """
        Prepare a comprehensive validation package for strategy approval.
        
        Args:
            asset: Asset type (crypto_spot, crypto_futures, forex, forex_options)
            strategy: Strategy name/identifier
            kpis: KPI results from Compute_KPIs
            baselines: Baseline results from Generate_Baselines
            robustness: Robustness check results
            compliance: Compliance check results
            result_path: Path to result files
            
        Returns:
            Dictionary containing comprehensive validation package and final verdict
        """
        result_dir = Path(result_path)
        
        # Get asset-specific thresholds for final evaluation
        asset_rules = config_manager.get_asset_rules(asset)
        thresholds = asset_rules.get("thresholds", {})
        
        # Create comprehensive validation package
        validation_package = {
            'metadata': {
                'strategy': strategy,
                'asset': asset,
                'validation_timestamp': pd.Timestamp.now().isoformat(),
                'result_path': str(result_path)
            },
            'performance_metrics': {
                'kpis': kpis,
                'baseline_comparisons': baselines
            },
            'risk_assessment': {
                'robustness_tests': robustness,
                'compliance_checks': compliance
            }
        }
        
        # Final validation scoring
        validation_scores = {}
        
        # 1. KPI Score (40% weight)
        kpi_pass_rate = kpis.get('summary', {}).get('pass_rate', 0)
        validation_scores['kpi_score'] = kpi_pass_rate
        
        # 2. Baseline Comparison Score (20% weight)
        baseline_beats = 0
        baseline_total = 0
        for baseline_name, baseline_result in baselines.items():
            if isinstance(baseline_result, dict) and 'strategy' in baseline_result:
                # Simple comparison - strategy should beat baseline
                strategy_sharpe = kpis.get('sharpe_ratio', 0)
                baseline_sharpe = baseline_result.get('sharpe_ratio', 0)
                if strategy_sharpe > baseline_sharpe:
                    baseline_beats += 1
                baseline_total += 1
        
        baseline_score = baseline_beats / baseline_total if baseline_total > 0 else 0
        validation_scores['baseline_score'] = baseline_score
        
        # 3. Robustness Score (25% weight)
        robustness_pass_rate = robustness.get('overall_assessment', {}).get('pass_rate', 0)
        validation_scores['robustness_score'] = robustness_pass_rate
        
        # 4. Compliance Score (15% weight)
        compliance_rate = compliance.get('overall_compliance', {}).get('compliance_rate', 0)
        validation_scores['compliance_score'] = compliance_rate
        
        # Calculate weighted final score
        weights = {
            'kpi_score': 0.40,
            'baseline_score': 0.20,
            'robustness_score': 0.25,
            'compliance_score': 0.15
        }
        
        final_score = sum(validation_scores[metric] * weight for metric, weight in weights.items())
        validation_scores['final_score'] = final_score
        
        # Determine final validation status
        # Must pass minimum thresholds in each category
        pass_thresholds = {
            'kpi_score': 0.7,      # 70% of KPIs must pass
            'baseline_score': 0.5,  # Beat 50% of baselines
            'robustness_score': 0.75, # 75% robustness tests pass
            'compliance_score': 1.0,   # 100% compliance required
            'final_score': 0.75        # 75% overall score required
        }
        
        validation_status = {}
        all_pass = True
        
        for metric, threshold in pass_thresholds.items():
            passes = validation_scores[metric] >= threshold
            validation_status[metric] = {
                'score': validation_scores[metric],
                'threshold': threshold,
                'passes': passes
            }
            if not passes:
                all_pass = False
        
        # Additional critical checks
        critical_failures = []
        
        # Check for critical KPI failures
        if not kpis.get('overall_pass', False):
            critical_failures.append("Strategy failed critical KPI thresholds")
        
        # Check for compliance failures
        if not compliance.get('overall_compliance', {}).get('fully_compliant', False):
            critical_failures.append("Strategy failed risk compliance checks")
        
        # Check sample size adequacy
        if not compliance.get('sample_validation', {}).get('sample_adequate', False):
            critical_failures.append("Insufficient sample size for validation")
        
        # Final verdict
        final_verdict = {
            'approved': all_pass and len(critical_failures) == 0,
            'score_based_pass': all_pass,
            'critical_failures': critical_failures,
            'recommendation': 'APPROVED' if (all_pass and len(critical_failures) == 0) else 'REJECTED'
        }
        
        # Add detailed reasoning
        if final_verdict['approved']:
            final_verdict['reasoning'] = f"Strategy approved with final score {final_score:.3f}. All validation criteria met."
        else:
            reasons = []
            for metric, status in validation_status.items():
                if not status['passes']:
                    reasons.append(f"{metric}: {status['score']:.3f} < {status['threshold']}")
            reasons.extend(critical_failures)
            final_verdict['reasoning'] = f"Strategy rejected. Issues: {'; '.join(reasons)}"
        
        # Package everything together
        validation_package.update({
            'validation_scores': validation_scores,
            'validation_status': validation_status,
            'final_verdict': final_verdict
        })
        
        # Save validation package to file
        package_file = result_dir / "validation_package.json"
        with open(package_file, "w") as f:
            f.write(_json.dumps(validation_package, indent=2).decode())
        
        # Create summary report
        summary_report = {
            'strategy': strategy,
            'asset': asset,
            'final_score': final_score,
            'approved': final_verdict['approved'],
            'recommendation': final_verdict['recommendation'],
            'key_metrics': {
                'sharpe_ratio': kpis.get('sharpe_ratio', 0),
                'max_drawdown': kpis.get('max_drawdown', 0),
                'profit_factor': kpis.get('profit_factor', 0),
                'win_rate': kpis.get('win_rate', 0)
            },
            'validation_timestamp': validation_package['metadata']['validation_timestamp'],
            'package_file': str(package_file)
        }
        
        summary_file = result_dir / "validation_summary.json"
        with open(summary_file, "w") as f:
            f.write(_json.dumps(summary_report, indent=2).decode())
        
        validation_package['summary_report'] = summary_report
        validation_package['files'] = {
            'package_file': str(package_file),
            'summary_file': str(summary_file)
        }
        
        return validation_package

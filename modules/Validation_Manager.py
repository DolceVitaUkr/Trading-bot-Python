import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

class ValidationManager:
    """
    Handles backtesting, walk-forward testing, and approval of trading strategies.
    Ensures only profitable and robust strategies are deployed.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the ValidationManager.

        Args:
            config (Dict[str, Any]): Configuration for the validation manager.
        """
        self.config = config
        self.min_trades = config.get("min_trades_for_approval", 500)
        self.min_sharpe = config.get("min_sharpe_ratio", 1.0)
        self.min_oos_sharpe = config.get("min_out_of_sample_sharpe", 0.5)

        self.approved_strategies: Dict[str, list] = {}  # {asset_class: [strategy_id, ...]}
        self.rejected_strategies: Dict[str, list] = {}  # {asset_class: [rejection_info, ...]}

        # For simulation, pre-populate with some mock validation results
        self._run_initial_validations()

    def _run_initial_validations(self):
        """Runs a set of mock validations at startup for simulation purposes."""
        logger.info("Running initial strategy validations for simulation...")

        # Scenario 1: Approved strategy
        self.run_validation_for_strategy(
            strategy_id="TrendFollowStrategy", asset_class="SPOT",
            mock_backtest_results={"trades": 600, "sharpe_ratio": 1.6, "max_drawdown": 0.12},
            mock_wfa_results={"trades": 250, "out_of_sample_sharpe": 1.2}
        )

        # Scenario 2: Rejected strategy (low trades)
        self.run_validation_for_strategy(
            strategy_id="MeanReversionStrategy", asset_class="SPOT",
            mock_backtest_results={"trades": 450, "sharpe_ratio": 1.8, "max_drawdown": 0.08},
            mock_wfa_results={"trades": 150, "out_of_sample_sharpe": 1.5}
        )

        # Scenario 3: Approved for PERP, but not for SPOT
        self.run_validation_for_strategy(
            strategy_id="MeanReversionStrategy", asset_class="PERP",
            mock_backtest_results={"trades": 800, "sharpe_ratio": 2.1, "max_drawdown": 0.07},
            mock_wfa_results={"trades": 300, "out_of_sample_sharpe": 1.7}
        )

    def run_validation_for_strategy(self, strategy_id: str, asset_class: str,
                                    mock_backtest_results: Dict, mock_wfa_results: Dict):
        """
        Orchestrates the validation process for a single strategy and asset class.
        In a real system, this would trigger full backtests. Here, it uses mock results.
        """
        logger.info(f"Validating '{strategy_id}' for asset class '{asset_class}'...")

        # In a real system, you would run the tests here.
        # backtest_results = self.backtest(strategy_id, asset_class)
        # wfa_results = self.walk_forward_test(strategy_id, asset_class)
        backtest_results = mock_backtest_results
        wfa_results = mock_wfa_results

        is_valid, reason = self._validate_results(backtest_results, wfa_results)

        if is_valid:
            self._approve(strategy_id, asset_class, {"backtest": backtest_results, "wfa": wfa_results})
        else:
            self._reject(strategy_id, asset_class, {"backtest": backtest_results, "wfa": wfa_results}, reason)

    def _validate_results(self, backtest: Dict, wfa: Dict) -> Tuple[bool, str]:
        """
        Checks if the combined results from backtesting and WFA meet the approval criteria.
        """
        total_trades = backtest.get("trades", 0)
        if total_trades < self.min_trades:
            return False, f"Insufficient trades ({total_trades} < {self.min_trades})"

        sharpe = backtest.get("sharpe_ratio", 0)
        if sharpe < self.min_sharpe:
            return False, f"Sharpe ratio too low ({sharpe:.2f} < {self.min_sharpe:.2f})"

        oos_sharpe = wfa.get("out_of_sample_sharpe", 0)
        if oos_sharpe < self.min_oos_sharpe:
            return False, f"Out-of-sample Sharpe ratio too low ({oos_sharpe:.2f} < {self.min_oos_sharpe:.2f})"

        return True, "Validation passed"

    def _approve(self, strategy_id: str, asset_class: str, results: Dict):
        """Adds a strategy to the approved list."""
        logger.info(f"Approving strategy '{strategy_id}' for asset class '{asset_class}'.",
                    extra={'action': 'strategy_approved', 'strategy_id': strategy_id,
                           'asset_class': asset_class, 'results': results})
        if asset_class not in self.approved_strategies:
            self.approved_strategies[asset_class] = []
        if strategy_id not in self.approved_strategies[asset_class]:
            self.approved_strategies[asset_class].append(strategy_id)

    def _reject(self, strategy_id: str, asset_class: str, results: Dict, reason: str):
        """Adds a strategy to the rejected list and logs it."""
        logger.warning(f"Rejecting strategy '{strategy_id}' for asset class '{asset_class}'. Reason: {reason}",
                       extra={'action': 'strategy_rejected', 'strategy_id': strategy_id,
                              'asset_class': asset_class, 'reason': reason, 'results': results})
        if asset_class not in self.rejected_strategies:
            self.rejected_strategies[asset_class] = []
        self.rejected_strategies[asset_class].append({
            "strategy_id": strategy_id,
            "reason": reason,
            "results": results,
            "timestamp": logging.Formatter().formatTime(logging.makeLogRecord({}))
        })

    def is_strategy_approved(self, strategy_id: str, asset_class: str) -> bool:
        """
        Checks if a strategy is currently in the approved list for a given asset class.

        Args:
            strategy_id (str): The strategy to check.
            asset_class (str): The asset class to check against.

        Returns:
            bool: True if the strategy is approved, False otherwise.
        """
        is_approved = strategy_id in self.approved_strategies.get(asset_class, [])
        logger.debug(f"Checking approval for '{strategy_id}' on '{asset_class}': {is_approved}")
        return is_approved

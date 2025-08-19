"""
Validation Manager to approve strategies based on backtest performance.
"""
import datetime
from typing import Any, Dict, Optional, Tuple

from trading_bot.core.interfaces import ValidationRunner
from trading_bot.core.schemas import ValidationRecord

try:
    import orjson as _json
except ImportError:
    import ujson as _json


# Constants for validation thresholds
MIN_TRADES = 500
MIN_SHARPE = 2.0
MAX_DD = 0.15
COOL_OFF_DAYS = 3
VALIDATION_FILE = "state/validation_runs.jsonl"


class ValidationManager(ValidationRunner):
    """
    Implements the validation logic for strategies, now based on product.
    """

    def __init__(self, validation_file: str = VALIDATION_FILE):
        self.validation_file = validation_file
        self._ensure_file_exists()
        # In a real application, trade counts would be persisted, e.g., in a database or a state file.
        # For this implementation, we'll use an in-memory dictionary.
        self.trade_counts: Dict[str, int] = {
            "FOREX_SPOT": 0,
            "FOREX_OPTIONS": 0,
            "CRYPTO_SPOT": 0,
        }

    def _ensure_file_exists(self):
        """Ensures the validation JSONL file exists."""
        try:
            with open(self.validation_file, "a"):
                pass
        except IOError:
            import os

            os.makedirs(os.path.dirname(self.validation_file), exist_ok=True)
            with open(self.validation_file, "a"):
                pass

    def _get_last_run(
        self, strategy_id: str, product: str
    ) -> Optional[ValidationRecord]:
        """
        Retrieves the last validation run for a given strategy and product.
        """
        last_run = None
        with open(self.validation_file, "rb") as f:
            for line in f:
                record = ValidationRecord.parse_raw(line)
                if record.strategy_id == strategy_id and record.product == product:
                    last_run = record
        return last_run

    def _append_record(self, record: ValidationRecord):
        """
        Appends a new validation record to the JSONL file.
        """
        with open(self.validation_file, "ab") as f:
            f.write(_json.dumps(record.dict(by_alias=True)))
            f.write(b"\n")

    def update_trade_count(self, product: str, count: int):
        """
        Updates the simulated trade count for a given product.
        This would be called by a training or simulation process.
        """
        self.trade_counts[product] = self.trade_counts.get(product, 0) + count
        print(f"Updated trade count for {product}: {self.trade_counts[product]}")

    async def approved(self, strategy_id: str, product: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Checks if a strategy is approved for live trading on a specific product.
        """
        # 1. Check for cool-off period
        last_run = self._get_last_run(strategy_id, product)
        if last_run and last_run.cool_off_until and datetime.datetime.now(datetime.timezone.utc) < last_run.cool_off_until:
            return False, {"reason": f"In cool-off period for {product} until {last_run.cool_off_until}"}

        # 2. Get simulated results (in a real bot, this would come from a backtesting engine or training pipeline)
        # We now use the in-memory trade count.
        simulated_results: Dict[str, Any] = {
            "n_trades": self.trade_counts.get(product, 0),
            "sharpe": 2.5, # Placeholder
            "max_dd": 0.10, # Placeholder
            "winrate": 0.6, # Placeholder
            "avg_pnl": 150.0, # Placeholder
            "slippage_mean": 0.01, # Placeholder
        }

        # 3. Check against thresholds
        reasons = []
        is_approved = True
        if simulated_results["n_trades"] < MIN_TRADES:
            is_approved = False
            reasons.append(f"Not enough trades for product {product}: {simulated_results['n_trades']} < {MIN_TRADES}")
        if simulated_results["sharpe"] < MIN_SHARPE:
            is_approved = False
            reasons.append(f"Sharpe ratio too low: {simulated_results['sharpe']} < {MIN_SHARPE}")
        if simulated_results["max_dd"] > MAX_DD:
            is_approved = False
            reasons.append(f"Max drawdown too high: {simulated_results['max_dd']} > {MAX_DD}")

        # 4. Create and persist the record
        decision = {"approved": is_approved, "reasons": reasons}
        now = datetime.datetime.now(datetime.timezone.utc)
        record = ValidationRecord(
            strategy_id=strategy_id,
            product=product,
            period="simulated",
            n_trades=simulated_results["n_trades"],
            sharpe=simulated_results["sharpe"],
            max_dd=simulated_results["max_dd"],
            winrate=simulated_results["winrate"],
            avg_pnl=simulated_results["avg_pnl"],
            slippage_mean=simulated_results["slippage_mean"],
            decision=decision,
            promoted_at=now if is_approved else None,
            cool_off_until=None if is_approved else now + datetime.timedelta(days=COOL_OFF_DAYS)
        )
        self._append_record(record)

        return is_approved, {"decision_record": record.dict()}

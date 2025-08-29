# file: tradingbot/core/configmanager.py
import os
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv

try:
    import orjson as _json
except ImportError:
    import ujson as _json


class ConfigManager:
    """Load and expose configuration data for the bot."""

    def __init__(self, config_dir: Path | None = None) -> None:
        load_dotenv()  # Load .env file
        if config_dir is None:
            config_dir = Path(__file__).resolve().parent.parent / "config"
        self.config_dir = config_dir
        self.config: Dict[str, Any] = self._load_json(config_dir / "config.json")
        self.assets: Dict[str, Any] = self._load_json(config_dir / "assets.json")
        self.strategies: Dict[str, Any] = self._load_json(
            config_dir / "strategies.json"
        )
        # Provide defaults for important safety keys if they are missing
        self.config.setdefault(
            "safety",
            {
                "START_MODE": "paper",
                "LIVE_TRADING_ENABLED": False,
                "REQUIRE_MANUAL_LIVE_CONFIRMATION": True,
                "ORDER_ROUTING": "paper_first",
                "MAX_STOP_LOSS_PCT": 0.15,
                "KILL_SWITCH_ENABLED": True,
                "CONSECUTIVE_LOSS_KILL": 3,
                "PAPER_EQUITY_START": 0.0,
                "PAPER_RESET_THRESHOLD": 0.0,
            },
        )
        
        # Add asset-specific rules matrix for multi-asset validation
        self.asset_rules = self._get_asset_rules_matrix()

    def _substitute_env_vars(self, config_value: Any) -> Any:
        """Recursively substitutes environment variables in config values."""
        if isinstance(config_value, dict):
            return {k: self._substitute_env_vars(v) for k, v in config_value.items()}
        elif isinstance(config_value, list):
            return [self._substitute_env_vars(v) for v in config_value]
        elif (
            isinstance(config_value, str)
            and config_value.startswith("_ENV_")
            and config_value.endswith("_")
        ):
            env_var_name = config_value.replace("_ENV_", "").strip("_")
            return os.getenv(env_var_name)
        return config_value

    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """Loads a JSON file, substitutes environment variables, and returns its content."""
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        with open(file_path, "rb") as f:
            data = _json.loads(f.read())
            return self._substitute_env_vars(data)

    def get_config(self) -> Dict[str, Any]:
        """Returns the main configuration."""
        return self.config

    def get_asset_config(self, asset_symbol: str) -> Dict[str, Any]:
        """Returns the configuration for a specific asset."""
        return self.assets.get(asset_symbol, {})

    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """Returns the configuration for a specific strategy."""
        return self.strategies.get(strategy_name, {})

    # Generic dotted-path getter -------------------------------------------------
    def get(self, path: str, default: Any | None = None) -> Any:
        """Retrieve a value from the main config using a dotted path."""
        node: Any = self.config
        for part in path.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def reload(self) -> None:
        """Reload configuration files from disk for all sections."""
        self.config = self._load_json(self.config_dir / "config.json")
        self.assets = self._load_json(self.config_dir / "assets.json")
        self.strategies = self._load_json(self.config_dir / "strategies.json")
        self.asset_rules = self._get_asset_rules_matrix()
        
    def _get_asset_rules_matrix(self) -> Dict[str, Any]:
        """Returns asset-specific rules matrix for multi-asset validation."""
        return {
            "crypto_spot": {
                "fees": {
                    "maker": 0.001,  # 0.1%
                    "taker": 0.001,  # 0.1%
                    "funding": 0.0   # No funding for spot
                },
                "slippage_model": "stochastic_spread_impact",
                "leverage_cap": 1,
                "lot_rules": {
                    "min_notional": 10.0,
                    "tick_size": 0.01,
                    "lot_size": 0.00001
                },
                "trading_hours": "24h",
                "funding_borrow": None,
                "min_sample": {
                    "trades": 1000,
                    "months": 6
                },
                "latency_jitter": 50,  # ms
                "execution_model": "limit_order_book",
                "risk_caps": {
                    "max_position_pct": 20,
                    "max_daily_trades": 100
                },
                "baselines": ["buy_hold", "vol_target_bh", "sma_10_20", "random_turnover_matched"],
                "thresholds": {
                    "sharpe": 2.0,
                    "sortino": 3.0,
                    "profit_factor": 1.8,
                    "max_dd": 0.15,
                    "cvar_95": 0.10,
                    "win_rate_min": 0.50,
                    "avg_win_loss_ratio": 1.5,
                    "expectancy": 0.20,
                    "avg_profit_trade_pct": 0.30,
                    "rolling_windows_pass": 0.75,
                    "seed_sharpe_spread": 0.3,
                    "oos_is_ratio": 0.7,
                    "pbo_max": 0.10,
                    "baseline_beat_sharpe": 1.2,
                    "baseline_beat_pf": 1.1
                }
            },
            "crypto_futures": {
                "fees": {
                    "maker": 0.0002,  # 0.02%
                    "taker": 0.0005,  # 0.05%
                    "funding": 0.0001  # Per 8 hours
                },
                "slippage_model": "stochastic_spread_impact_x2",
                "leverage_cap": 50,
                "lot_rules": {
                    "min_notional": 5.0,
                    "tick_size": 0.01,
                    "lot_size": 0.001
                },
                "trading_hours": "24h",
                "funding_borrow": "3x_daily",
                "min_sample": {
                    "trades": 1000,
                    "months": 6
                },
                "latency_jitter": 100,  # ms
                "execution_model": "perpetual_futures",
                "risk_caps": {
                    "max_position_pct": 10,
                    "max_daily_trades": 200,
                    "liquidation_buffer": 0.2
                },
                "baselines": ["buy_hold", "vol_target_bh", "sma_10_20", "random_turnover_matched"],
                "stress_tests": {
                    "funding_multiplier": 2,
                    "slippage_multiplier": 2
                },
                "thresholds": {
                    "sharpe": 2.0,
                    "sortino": 3.0,
                    "profit_factor": 1.8,
                    "max_dd": 0.15,
                    "cvar_95": 0.10,
                    "win_rate_min": 0.50,
                    "avg_win_loss_ratio": 1.5,
                    "expectancy": 0.20,
                    "avg_profit_trade_pct": 0.30,
                    "rolling_windows_pass": 0.75,
                    "seed_sharpe_spread": 0.3,
                    "oos_is_ratio": 0.7,
                    "pbo_max": 0.10,
                    "baseline_beat_sharpe": 1.2,
                    "baseline_beat_pf": 1.1
                }
            },
            "forex": {
                "fees": {
                    "spread": 0.0001,  # 1 pip for majors
                    "commission": 2.0   # USD per 100k
                },
                "slippage_model": "tight_spread_sessions",
                "leverage_cap": 50,
                "lot_rules": {
                    "standard_lot": 100000,
                    "min_lot": 0.01,
                    "tick_size": 0.00001
                },
                "trading_hours": "session_based",
                "funding_borrow": None,
                "min_sample": {
                    "trades": 1000,
                    "months": 6
                },
                "latency_jitter": 25,  # ms
                "execution_model": "ecn_spot_fx",
                "risk_caps": {
                    "max_position_pct": 15,
                    "max_daily_trades": 150
                },
                "baselines": ["carry_neutral_sma_50_200", "buy_hold_synthetic", "random"],
                "thresholds": {
                    "sharpe": 2.0,
                    "sortino": 3.0,
                    "profit_factor": 1.8,
                    "max_dd": 0.15,
                    "cvar_95": 0.10,
                    "win_rate_min": 0.50,
                    "avg_win_loss_ratio": 1.5,
                    "expectancy": 0.20,
                    "avg_profit_trade_pct": 0.30,
                    "rolling_windows_pass": 0.75,
                    "seed_sharpe_spread": 0.3,
                    "oos_is_ratio": 0.7,
                    "pbo_max": 0.10,
                    "baseline_beat_sharpe": 1.2,
                    "baseline_beat_pf": 1.1
                }
            },
            "forex_options": {
                "fees": {
                    "premium_pct": 0.001,  # 0.1% of premium
                    "commission": 1.0      # Per contract
                },
                "slippage_model": "mid_spread_fill_prob",
                "leverage_cap": 1,
                "lot_rules": {
                    "contract_size": 10000,
                    "min_premium": 0.0001,
                    "tick_size": 0.0001
                },
                "trading_hours": "session_based",
                "funding_borrow": None,
                "min_sample": {
                    "trades": 500,  # Lower for options
                    "months": 3
                },
                "latency_jitter": 200,  # ms
                "execution_model": "options_market_maker",
                "risk_caps": {
                    "max_position_pct": 25,
                    "max_daily_trades": 50,
                    "delta_band": 0.5,
                    "gamma_limit": 1000,
                    "vega_limit": 5000
                },
                "baselines": ["covered_call", "delta_hedged_straddle"],
                "thresholds": {
                    "sharpe": 1.8,  # Slightly lower for options
                    "sortino": 2.5,
                    "profit_factor": 1.6,
                    "max_dd": 0.20,  # Higher tolerance
                    "cvar_95": 0.15,
                    "win_rate_min": 0.45,  # Lower for options
                    "avg_win_loss_ratio": 1.3,
                    "expectancy": 0.15,
                    "avg_profit_trade_pct": 0.25,
                    "rolling_windows_pass": 0.75,
                    "seed_sharpe_spread": 0.4,
                    "oos_is_ratio": 0.65,
                    "pbo_max": 0.15,
                    "baseline_beat_sharpe": 1.1,
                    "baseline_beat_pf": 1.05
                }
            }
        }
        
    def get_asset_rules(self, asset_type: str) -> Dict[str, Any]:
        """Get asset-specific rules for validation."""
        return self.asset_rules.get(asset_type, {})


# Global instance for easy access
# This can be replaced with dependency injection later if needed
config_manager = ConfigManager()


def load_config() -> None:
    """Reload the global config manager instance from disk."""
    config_manager.reload()


def save_config() -> None:
    """Persist the current in-memory configuration back to disk."""
    with open(config_manager.config_dir / "config.json", "wb") as f:
        f.write(_json.dumps(config_manager.config))
    with open(config_manager.config_dir / "assets.json", "wb") as f:
        f.write(_json.dumps(config_manager.assets))
    with open(config_manager.config_dir / "strategies.json", "wb") as f:
        f.write(_json.dumps(config_manager.strategies))


def get_param(path: str, default: Any | None = None) -> Any:
    """Access a nested configuration value from the global instance."""
    return config_manager.get(path, default)
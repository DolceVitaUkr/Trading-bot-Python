# file: core/config_manager.py
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

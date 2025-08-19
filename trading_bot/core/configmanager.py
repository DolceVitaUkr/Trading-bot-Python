import os
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv

try:
    import orjson as _json
except ImportError:
    import ujson as _json


class ConfigManager:
    """
    Manages the bot's configuration by loading from multiple JSON files,
    with support for environment variable substitution.
    """

    def __init__(self, config_dir: Path = Path("trading_bot/config")):
        load_dotenv()  # Load .env file
        self.config_dir = config_dir
        self.config: Dict[str, Any] = self._load_json(config_dir / "config.json")
        self.assets: Dict[str, Any] = self._load_json(config_dir / "assets.json")
        self.strategies: Dict[str, Any] = self._load_json(
            config_dir / "strategies.json"
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


# Global instance for easy access
# This can be replaced with dependency injection later if needed
config_manager = ConfigManager()

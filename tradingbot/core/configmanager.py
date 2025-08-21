# file: core/configmanager.py
"""Compatibility wrapper for config_manager module."""

from .config_manager import (
    ConfigManager,
    config_manager,
    load_config,
    save_config,
    get_param,
)

__all__ = [
    "ConfigManager",
    "config_manager",
    "load_config",
    "save_config",
    "get_param",
]

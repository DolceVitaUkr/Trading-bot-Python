# file: tradingbot/core/tradeexecutor.py 
# module_version: v1.01

"""
DEPRECATED COMPATIBILITY WRAPPER
This file is deprecated. Use trade_executor.py or order_router.py directly.
This wrapper only exists for backward compatibility and will be removed.
"""

import warnings
from .trade_executor import TradeExecutor

# Issue deprecation warning
warnings.warn(
    "tradeexecutor.py is deprecated. Use trade_executor.py or order_router.py directly.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ["TradeExecutor"]

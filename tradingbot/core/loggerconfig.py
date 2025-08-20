import logging
import sys
import structlog
from structlog.types import Processor

def setup_logging(log_level: str = "INFO"):
    """
    Sets up structured logging using structlog.

    Logs to the console will be human-readable and colored.
    Logs to a file will be in JSON format (JSONL).
    """
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=shared_processors + [
            # This is the final processor that formats the log record.
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure the formatter for file logging (JSON)
    json_formatter = structlog.stdlib.ProcessorFormatter(
        # The "event" field is the main message.
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=shared_processors,
    )

    # Ensure logs directory exists
    try:
        import os
        os.makedirs('logs', exist_ok=True)
    except OSError:
        pass

    # Use a .jsonl extension to indicate JSON Lines format
    file_handler = logging.FileHandler("logs/structured_logs.jsonl", mode="a")
    file_handler.setFormatter(json_formatter)

    # Configure the formatter for console logging (human-readable)
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(colors=True),
        foreign_pre_chain=shared_processors,
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    # Get the root logger and add the handlers
    root_logger = logging.getLogger()
    # Clear existing handlers to avoid duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(log_level.upper())

    # Mute other noisy loggers to keep the output clean
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)

    log = structlog.get_logger("main_config")
    log.info("Logging configured", log_level=log_level, file_path="logs/structured_logs.jsonl")

def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Returns a structlog logger instance.
    """
    return structlog.get_logger(name)

# Example of how to use the logger with context:
#
# from tradingbot.core.loggerconfig import get_logger
#
# # Get a logger for the current module
# log = get_logger(__name__)
#
# # Bind context that will be included in all subsequent logs from this logger instance.
# # This is perfect for setting the product context at the start of a pipeline.
# product_log = log.bind(product="FOREX_SPOT", symbol="EURUSD")
#
# product_log.info("fetching_data", duration="1M", timeframe="1h")
# # This log record will contain:
# # {"product": "FOREX_SPOT", "symbol": "EURUSD", "event": "fetching_data",
# #  "duration": "1M", "timeframe": "1h", "timestamp": "...", ...}
#
# # You can also add context for a single call, which is great for metrics.
# product_log.info("request_finished", latency_ms=120, req_count=1, pacing_backoffs=0)
# # This log record will contain:
# # {"product": "FOREX_SPOT", "symbol": "EURUSD", "event": "request_finished",
# #  "latency_ms": 120, "req_count": 1, "pacing_backoffs": 0, "timestamp": "...", ...}

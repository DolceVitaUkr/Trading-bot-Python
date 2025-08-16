import logging
import json
from datetime import datetime

class CustomAdapter(logging.LoggerAdapter):
    """
    This adapter adds contextual information to the log record.
    """
    def process(self, msg, kwargs):
        # We merge the 'extra' dict into the kwargs for the formatter
        if 'extra' in kwargs:
            kwargs['extra']['extra_info'] = kwargs.pop('extra')
        return msg, kwargs

class JsonFormatter(logging.Formatter):
    """
    Formats log records as a JSON string.
    """
    def __init__(self, fields):
        super().__init__()
        self.fields = fields

    def format(self, record):
        # Create a dict from the specified fields
        log_record = {field: getattr(record, field, None) for field in self.fields}
        log_record['timestamp'] = datetime.fromtimestamp(record.created).isoformat()

        # Merge the extra dictionary
        if hasattr(record, 'extra_info'):
            for key, value in record.extra_info.items():
                log_record[key] = value

        return json.dumps(log_record)

def setup_logging(log_level=logging.INFO):
    """
    Sets up structured JSON logging to a file and standard logging to the console.
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    # Console Handler (human-readable)
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File Handler (JSON)
    log_fields = [
        'timestamp', 'levelname', 'name', 'module', 'funcName', 'lineno', 'message',
        'action', 'asset_class', 'session', 'symbol', 'mode', 'size_usd',
        'leverage', 'fees_usd', 'pnl_net_usd', 'allowed_funds', 'pair_cap_pct',
        'reason'
    ]
    # Ensure logs directory exists
    try:
        import os
        os.makedirs('logs', exist_ok=True)
    except OSError:
        pass

    file_handler = logging.FileHandler('logs/trading_bot.json', mode='a')
    json_formatter = JsonFormatter(fields=log_fields)
    file_handler.setFormatter(json_formatter)
    logger.addHandler(file_handler)

    logging.info("Logging configured with console and JSON file handlers.")

def get_logger(name: str):
    """
    Returns a logger instance.
    This is a convenience function to ensure all modules get the same logger setup.
    """
    return logging.getLogger(name)

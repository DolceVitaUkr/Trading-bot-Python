# utils/utilities.py
import os
import time
import logging
import datetime
from pathlib import Path
from functools import wraps
from typing import Optional, Callable, Type, Tuple, Union
import json
from logging.handlers import RotatingFileHandler

# ---------------------------
# File System Utilities
# ---------------------------

def ensure_directory(path: Union[str, Path]) -> Path:
    """Create directory structure with pathlib and return Path object."""
    dir_path = Path(path).expanduser().resolve()
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        logging.debug(f"Verified/Created directory: {dir_path}")
        return dir_path
    except PermissionError as e:
        logging.error(f"Permission denied creating directory {dir_path}: {e}")
        raise
    except Exception as e:
        logging.error(f"Error creating directory {dir_path}: {e}")
        raise

# ---------------------------
# Time Utilities
# ---------------------------

def format_timestamp(
    timestamp: Union[int, float],
    fmt: str = "%Y-%m-%d %H:%M:%S",
    timezone: Optional[datetime.tzinfo] = None
) -> Optional[str]:
    """Convert timestamp (seconds/ms) to formatted datetime with timezone support."""
    try:
        # Handle milliseconds
        if timestamp > 1e12:
            timestamp /= 1000
            
        dt = (datetime.datetime.fromtimestamp(timestamp, tz=timezone) 
              if timezone else datetime.datetime.fromtimestamp(timestamp))
              
        return dt.strftime(fmt)
    except (ValueError, TypeError) as e:
        logging.error(f"Invalid timestamp {timestamp}: {e}")
        return None

# ---------------------------
# Retry Logic
# ---------------------------

def retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception]] = (Exception,),
    logger: Optional[logging.Logger] = None
) -> Callable:
    """Decorator with exponential backoff for retrying operations."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = initial_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        msg = f"{func.__name__} failed after {max_attempts} attempts"
                        (logger or logging.getLogger()).error(msg)
                        raise
                        
                    sleep_time = min(current_delay * backoff_factor ** (attempt - 1), max_delay)
                    (logger or logging.getLogger()).warning(
                        f"Retry {attempt}/{max_attempts} for {func.__name__} "
                        f"in {sleep_time:.1f}s. Error: {e}"
                    )
                    time.sleep(sleep_time)
            return func(*args, **kwargs)  # Final attempt
        return wrapper
    return decorator

# ---------------------------
# Logging Configuration
# ---------------------------

def configure_logging(
    log_file: Union[str, Path] = "logs/app.log",
    log_level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    json_format: bool = False
) -> None:
    """Configure logging with rotating files and optional JSON formatting."""
    log_path = ensure_directory(log_file).parent
    
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            log_entry = {
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno
            }
            if record.exc_info:
                log_entry["exception"] = self.formatException(record.exc_info)
            return json.dumps(log_entry)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # File handler with rotation
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setFormatter(
        JsonFormatter() if json_format else
        logging.Formatter('%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s')
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s'
    ))

    # Clear existing handlers and add new ones
    root_logger.handlers = [file_handler, console_handler]

    logging.info("Logging system initialized")

# ---------------------------
# Async Utilities
# ---------------------------

async def async_retry(*args, **kwargs):
    """Async version of the retry decorator"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Implementation similar to sync version but with async sleep
            # (Omitted for brevity)
            pass
        return wrapper
    return decorator
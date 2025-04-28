# utils/utilities.py

import os
import logging
import functools
import time
from datetime import datetime, date
from typing import Any, Callable, Tuple, Type, Union

def ensure_directory(path: str) -> None:
    """
    Create `path` (and any parent dirs) if it doesn’t already exist,
    handling race conditions safely.
    """
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        if not os.path.isdir(path):
            raise

def configure_logging(
    level: Union[int, str] = logging.INFO,
    log_file: str = None,
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> None:
    """
    Configure the root logger.  
    If `log_file` is given, logs to that file, otherwise to stderr.
    """
    logger = logging.getLogger()
    # allow string levels like 'DEBUG'
    lvl = level if isinstance(level, int) else getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(lvl)

    handler = logging.FileHandler(log_file) if log_file else logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))
    # avoid adding multiple handlers if re-called
    if not any(isinstance(h, type(handler)) for h in logger.handlers):
        logger.addHandler(handler)

def format_timestamp(ts: Union[int, float, datetime, date]) -> str:
    """
    Turn a UNIX timestamp (seconds or ms), datetime or date into
    a UTC ISO-8601 string.
    """
    if isinstance(ts, (int, float)):
        # assume ms if >1e12
        secs = ts / 1000 if ts > 1e12 else ts
        dt = datetime.utcfromtimestamp(secs)
    elif isinstance(ts, date):
        dt = datetime.combine(ts, datetime.min.time())
    elif isinstance(ts, datetime):
        dt = ts
    else:
        raise TypeError(f"Cannot format timestamp of type {type(ts)}")
    return dt.replace(tzinfo=None).isoformat() + "Z"

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
    backoff: float = 2.0
) -> Callable:
    """
    Decorator to retry a function up to `max_attempts` times on the given
    exception types, waiting `delay` seconds (then `delay*backoff`, etc.).
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            m, d = max_attempts, delay
            last_exc = None
            for attempt in range(1, m + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    logging.warning(f"[retry] {fn.__name__} failed on attempt {attempt}/{m}: {e}")
                    if attempt == m:
                        logging.error(f"[retry] All {m} attempts failed for {fn.__name__}")
                        raise
                    time.sleep(d)
                    d *= backoff
            # unreachable
            raise last_exc
        return wrapper
    return decorator

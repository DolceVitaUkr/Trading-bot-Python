import functools
import io
import json
import logging
import os
import time
from datetime import date, datetime, timezone
from typing import (Any, Callable, Iterable, Iterator, List, Optional, Tuple,
                    Type, Union)

# Import the new logging setup function
from trading_bot.core.loggerconfig import setup_logging

# ────────────────────────────────────────────────────────────────────────────────
# Filesystem helpers
# ────────────────────────────────────────────────────────────────────────────────


def ensure_directory(path: str) -> None:
    """
    Create `path` if it doesn’t already exist.
    """
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        if not os.path.isdir(path):
            raise


def write_json(
    path: str, data: Any, atomic: bool = True, indent: int = 2
) -> None:
    """
    Safely write JSON to disk.
    """
    ensure_directory(os.path.dirname(path) or ".")
    payload = json.dumps(data, indent=indent, ensure_ascii=False)
    if atomic:
        tmp_path = f"{path}.tmp"
        with io.open(tmp_path, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    else:
        with io.open(path, "w", encoding="utf-8") as f:
            f.write(payload)


def read_json(path: str, default: Any = None) -> Any:
    """
    Read JSON from disk; returns `default` on missing file or parse error.
    """
    try:
        with io.open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except Exception as e:
        logging.warning(f"read_json: failed to read {path}: {e}")
        return default


# ────────────────────────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────────────────────────

# The configure_logging function has been removed.
# Directy use setup_logging from trading_bot.core.loggerconfig instead.


# ────────────────────────────────────────────────────────────────────────────────
# Time helpers
# ────────────────────────────────────────────────────────────────────────────────

def utc_now() -> datetime:
    """Timezone-aware utcnow()."""
    return datetime.now(timezone.utc)


def format_timestamp(ts: Union[int, float, datetime, date]) -> str:
    """
    Turn a timestamp into a UTC ISO-8601 string.
    """
    if isinstance(ts, (int, float)):
        secs = ts / 1000 if ts > 1e12 else ts
        dt = datetime.fromtimestamp(secs, tz=timezone.utc)
    elif isinstance(ts, date) and not isinstance(ts, datetime):
        dt = datetime.combine(ts, datetime.min.time(), tzinfo=timezone.utc)
    elif isinstance(ts, datetime):
        dt = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    else:
        raise TypeError(f"Cannot format timestamp of type {type(ts)}")
    return dt.astimezone(timezone.utc).isoformat()


# ────────────────────────────────────────────────────────────────────────────────
# Retry decorator
# ────────────────────────────────────────────────────────────────────────────────

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
    backoff: float = 2.0,
    logger: Optional[logging.Logger] = None,
) -> Callable:
    """
    Decorator to retry a function up to `max_attempts` times.
    """
    log = logger or logging.getLogger(__name__)

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            d = delay
            last_exc: Optional[BaseException] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    log.warning(
                        f"[retry] {fn.__name__} failed "
                        f"({attempt}/{max_attempts}): {e}")
                    if attempt >= max_attempts:
                        log.error(
                            f"[retry] Exhausted attempts for {fn.__name__}")
                        raise
                    time.sleep(d)
                    d *= backoff
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator


# ────────────────────────────────────────────────────────────────────────────────
# Small utils
# ────────────────────────────────────────────────────────────────────────────────

def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp value into [min_value, max_value]."""
    return max(min_value, min(value, max_value))


def chunked(iterable: Iterable[Any], n: int) -> Iterator[List[Any]]:
    """
    Yield lists of size n from an iterable.
    """
    chunk: List[Any] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= n:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

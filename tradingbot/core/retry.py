"""
Simple retry/backoff helpers.
"""
from __future__ import annotations
import time, random
from typing import Callable, Tuple, Type

def retry_call(func: Callable, *args, retries: int = 3, backoff: float = 0.5, max_backoff: float = 4.0, exceptions: Tuple[Type[BaseException], ...] = (Exception,), jitter: bool = True, **kwargs):
    delay = backoff
    last = None
    for attempt in range(retries + 1):
        try:
            return func(*args, **kwargs)
        except exceptions as e:
            last = e
            if attempt >= retries:
                break
            sleep = delay + (random.uniform(0, delay/4) if jitter else 0.0)
            time.sleep(sleep)
            delay = min(delay * 2.0, max_backoff)
    raise last
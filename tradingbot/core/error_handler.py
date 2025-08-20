"""Centralised error handling (minimal placeholder)."""

from __future__ import annotations

import logging
from typing import Callable


class ErrorHandler:
    def __init__(self) -> None:
        self.log = logging.getLogger(__name__)

    def wrap(self, func: Callable) -> Callable:
        def inner(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - logging side effects
                self.log.exception("Unhandled error: %s", exc)
                raise
        return inner


__all__ = ["ErrorHandler"]

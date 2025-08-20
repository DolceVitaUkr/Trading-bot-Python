"""Simplified notification helper used in tests.

The real project integrates Telegram alerts and rich formatting.  For the
unit tests we only need a minimal sink that records messages so that the
runtime controller can notify about important events.
"""

from __future__ import annotations

import logging
from typing import List


class Notifier:
    """Collects messages and logs them."""

    def __init__(self) -> None:
        self.messages: List[str] = []
        self.log = logging.getLogger(__name__)

    def send(self, message: str) -> None:
        self.messages.append(message)
        self.log.info(message)


__all__ = ["Notifier"]

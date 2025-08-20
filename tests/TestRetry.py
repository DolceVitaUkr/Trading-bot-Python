import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from TradingBot.core.utilities import retry


def test_retry_eventually_succeeds():
    attempts = {"count": 0}

    @retry(max_attempts=3, delay=0)
    def flaky():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise ValueError("fail")
        return "success"

    assert flaky() == "success"
    assert attempts["count"] == 3


def test_retry_raises_after_exhaustion():
    attempts = {"count": 0}

    @retry(max_attempts=3, delay=0)
    def always_fail():
        attempts["count"] += 1
        raise ValueError("fail")

    with pytest.raises(ValueError):
        always_fail()
    assert attempts["count"] == 3

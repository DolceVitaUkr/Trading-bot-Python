import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from TradingBot.__main__ import main


def test_main_callable():
    """Ensure the entrypoint main function is callable."""
    assert callable(main)

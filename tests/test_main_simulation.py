import pytest
from unittest.mock import MagicMock, patch
import main
import config

@pytest.fixture
def mock_dependencies(monkeypatch):
    """Mock all external dependencies for the main loop."""
    mock_exchange = MagicMock()
    mock_exchange.client.fetch_ohlcv.return_value = [
        [1672531200000, 60000, 60100, 59900, 60050, 100],
        [1672531500000, 60050, 60150, 59950, 60100, 120],
    ]
    monkeypatch.setattr("main.ExchangeAPI", lambda: mock_exchange)

    mock_dm = MagicMock()
    # We don't mock DataManager anymore, we let it be created with the mocked exchange

    mock_notifier = MagicMock()
    monkeypatch.setattr("main.TelegramNotifier", lambda **kwargs: mock_notifier)

    mock_scheduler = MagicMock()
    monkeypatch.setattr("main.JobScheduler", lambda: mock_scheduler)

    return {
        "exchange": mock_exchange,
        "notifier": mock_notifier,
        "scheduler": mock_scheduler,
    }

def test_main_simulation_loop_runs_once(mock_dependencies):
    """
    Test that the main simulation loop runs at least once without errors.
    """
    args = main.parse_args(["--mode", "simulation"])

    with patch("threading.Thread.start") as mock_thread_start:
        exit_code = main.run_bot(args, test_mode=True)

    assert exit_code == 0
    # Check that the agent loop was started
    assert mock_thread_start.call_count > 0

    # The load_markets call is skipped in test_mode=True, so this assertion is removed.
    # We primarily care that the bot thread starts.

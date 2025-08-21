import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from tradingbot.__main__ import main
from tradingbot.core.runtimecontroller import RuntimeController


def test_main_callable():
    assert callable(main)


def test_runtime_starts_in_paper(tmp_path):
    rc = RuntimeController(state_path=tmp_path / "runtime.json")
    state = rc.getstate()
    assert state["global"]["kill_switch"] is False
    assert state["assets"] == {}

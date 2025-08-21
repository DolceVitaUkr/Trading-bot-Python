import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from tradingbot.core.pairmanager import PairManager
from tradingbot.learning.trainmlmodel import (
    purgedtraintestsplit,
    triplebarrierlabel,
    trainml,
    loadml,
    predictml,
)
from tradingbot.learning.trainrlmodel import trainrl, loadrl, evaluaterl
from tradingbot.learning.statefeaturizer import buildstate


def test_triple_barrier_labels():
    prices = pd.Series([100, 103, 98, 105])
    labels = triplebarrierlabel(prices, 0.02, 0.02, 3)
    assert list(labels) == [1, -1, 1, 0]


def test_purged_split_applies_embargo():
    train, test = purgedtraintestsplit(100, test_size=0.2, embargo_pct=0.05)
    # last 5 observations of train removed due to embargo
    assert train[-1] == 74  # 80 - 5 - 1
    assert test[0] == 80


def test_trainml_and_predictml(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 3, 4, 5]})
    y = pd.Series([0, 0, 1, 1])
    path = trainml(df, y)
    model = loadml(path)
    preds = predictml(model, df)
    assert list(preds) == [0, 0, 1, 1]


def test_trainrl_and_evaluaterl():
    def stream():
        import random
        for _ in range(50):
            state = 0
            action = random.randint(0, 1)
            next_state = 1 if action == 1 else 0
            yield state, action, next_state, False

    def reward(state, action, next_state):
        return 1.0 if state == 0 and action == 1 else 0.0

    path = trainrl(stream(), reward, {"actions": 2, "lr": 0.5})
    model = loadrl(path)
    def eval_stream():
        for _ in range(10):
            yield 0, 0, 1, False
    total = evaluaterl(eval_stream(), model, reward)
    assert total == 10.0


def test_buildstate_creates_features():
    df = pd.DataFrame({"price": [1, 2, 3, 4, 5]})
    out = buildstate(df, {"price": 2})
    assert "price_ma2" in out.columns


def test_pair_manager_tag_regimes():
    df = pd.DataFrame(
        {
            "close": [100, 101, 102, 103, 104, 105],
            "volume": [2e6] * 6,
        }
    )
    pm = PairManager()
    regimes = pm.tagregimes(df)
    assert regimes["volatility"] in {"LOW", "HIGH"}
    assert regimes["trend"] == "UP"
    assert regimes["liquidity"] == "HIGH"


def test_pair_manager_refresh_and_top():
    pm = PairManager(default=["AAA", "BBB", "CCC"])
    pm.setsentiment(lambda sym: 1.0 if sym == "BBB" else 0.0)
    uni = pm.refreshuniverse()
    assert uni["crypto"][0] == "BBB"
    assert pm.gettop(2, "crypto") == uni["crypto"][:2]

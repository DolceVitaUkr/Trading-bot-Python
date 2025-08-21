import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from tradingbot.core.pair_manager import PairManager
from tradingbot.learning.train_ml_model import (
    purged_train_test_split,
    triple_barrier_label,
)


def test_triple_barrier_labels():
    prices = pd.Series([100, 103, 98, 105])
    labels = triple_barrier_label(prices, 0.02, 0.02, 3)
    assert list(labels) == [1, -1, 1, 0]


def test_purged_split_applies_embargo():
    train, test = purged_train_test_split(100, test_size=0.2, embargo_pct=0.05)
    # last 5 observations of train removed due to embargo
    assert train[-1] == 74  # 80 - 5 - 1
    assert test[0] == 80


def test_pair_manager_tag_regimes():
    df = pd.DataFrame(
        {
            "close": [100, 101, 102, 103, 104, 105],
            "volume": [2e6] * 6,
        }
    )
    pm = PairManager()
    regimes = pm.tag_regimes(df)
    assert regimes["volatility"] in {"LOW", "HIGH"}
    assert regimes["trend"] == "UP"
    assert regimes["liquidity"] == "HIGH"


def test_pair_manager_refresh_and_top():
    pm = PairManager(default=["AAA", "BBB", "CCC"])
    pm.set_sentiment(lambda sym: 1.0 if sym == "BBB" else 0.0)
    uni = pm.refresh_universe()
    assert uni["crypto"][0] == "BBB"
    assert pm.get_top(2, "crypto") == uni["crypto"][:2]

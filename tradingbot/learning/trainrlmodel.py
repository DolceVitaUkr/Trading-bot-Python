# file: learning/trainrlmodel.py
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Tuple
from uuid import uuid4
import pickle

MODELS_DIR = Path(__file__).resolve().parents[1] / "state" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def trainrl(datastream: Iterable[Tuple[int, int, int, bool]], rewardfn: Callable[[int, int, int], float], config: dict | None = None) -> str:
    """Train a tiny tabular Q-learning agent and persist it."""
    cfg = {"lr": 0.1, "gamma": 0.95, "actions": 2}
    if config:
        cfg.update(config)
    q: dict[tuple[int, int], float] = defaultdict(float)
    actions = cfg["actions"]
    for state, action, next_state, done in datastream:
        reward = rewardfn(state, action, next_state)
        max_next = 0.0 if done else max(q[(next_state, a)] for a in range(actions))
        q[(state, action)] += cfg["lr"] * (reward + cfg["gamma"] * max_next - q[(state, action)])
    path = MODELS_DIR / f"rl_{uuid4().hex}.pkl"
    with open(path, "wb") as f:
        pickle.dump({"q": dict(q), "actions": actions}, f)
    return str(path)

def loadrl(path: str) -> Any:
    """Load a saved RL model."""
    with open(path, "rb") as f:
        return pickle.load(f)

def evaluaterl(datastream: Iterable[Tuple[int, int, int, bool]], model: Any, rewardfn: Callable[[int, int, int], float]) -> float:
    """Evaluate a Q-table on a datastream."""
    q = model["q"]
    actions = model["actions"]
    total = 0.0
    for state, _action, next_state, done in datastream:
        best_action = max(range(actions), key=lambda a: q.get((state, a), 0.0))
        total += rewardfn(state, best_action, next_state)
        if done:
            continue
    return float(total)

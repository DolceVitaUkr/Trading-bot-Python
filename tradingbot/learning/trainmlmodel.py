# file: learning/trainmlmodel.py
from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4
import pickle

import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parents[1] / "state" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def triplebarrierlabel(prices: pd.Series, upper_pct: float, lower_pct: float, max_holding: int) -> pd.Series:
    """Label price series using the triple-barrier method.

    Returns a series with 1 for upper barrier, -1 for lower and 0 otherwise.
    """
    prices = prices.reset_index(drop=True)
    n = len(prices)
    labels = np.zeros(n, dtype=int)
    for i in range(n):
        start = prices[i]
        upper = start * (1 + upper_pct)
        lower = start * (1 - lower_pct)
        end = min(n, i + max_holding + 1)
        for j in range(i + 1, end):
            p = prices[j]
            if p >= upper:
                labels[i] = 1
                break
            if p <= lower:
                labels[i] = -1
                break
    return pd.Series(labels, index=prices.index)

def purgedtraintestsplit(n_samples: int, test_size: float = 0.2, embargo_pct: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Return train/test indices applying an optional embargo."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    test_start = int(n_samples * (1 - test_size))
    embargo = int(n_samples * embargo_pct)
    train_end = max(0, test_start - embargo)
    train_idx = np.arange(0, train_end)
    test_idx = np.arange(test_start, n_samples)
    return train_idx, test_idx

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def trainml(features: pd.DataFrame, labels: pd.Series, config: dict | None = None) -> str:
    """Train a simple logistic regression using gradient descent and persist it."""
    cfg = {"lr": 0.1, "epochs": 200}
    if config:
        cfg.update(config)
    X = features.to_numpy(dtype=float)
    y = labels.to_numpy(dtype=float)
    X = np.c_[np.ones(len(X)), X]
    w = np.zeros(X.shape[1])
    for _ in range(cfg["epochs"]):
        preds = _sigmoid(X @ w)
        grad = X.T @ (preds - y) / len(X)
        w -= cfg["lr"] * grad
    path = MODELS_DIR / f"ml_{uuid4().hex}.pkl"
    with open(path, "wb") as f:
        pickle.dump(w, f)
    return str(path)


def loadml(path: str) -> Any:
    """Load a previously saved weight vector."""
    with open(path, "rb") as f:
        return pickle.load(f)


def predictml(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Return binary predictions for the given feature matrix."""
    X = features.to_numpy(dtype=float)
    X = np.c_[np.ones(len(X)), X]
    probs = _sigmoid(X @ model)
    return (probs >= 0.5).astype(int)

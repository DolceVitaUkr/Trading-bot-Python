# file: learning/statefeaturizer.py
from __future__ import annotations

from typing import Dict
import pandas as pd

def buildstate(df: pd.DataFrame, spec: Dict[str, int]) -> pd.DataFrame:
    """Construct a simple feature set based on rolling means.

    ``spec`` maps column names to window sizes.
    """
    features = df.copy()
    for col, window in spec.items():
        features[f"{col}_ma{window}"] = features[col].rolling(window).mean()
    features.dropna(inplace=True)
    return features

# file: core/feature_store.py
"""Feature store utilities for persisting and validating feature DataFrames."""

from pathlib import Path

import pandas as pd
from jsonschema import validate as jsonschema_validate


def _feature_path(symbol: str, version: str, base_path: Path | None = None) -> Path:
    """Construct the filesystem path for a feature set."""
    base = base_path or Path(__file__).resolve().parent.parent / "state"
    return base / "features" / symbol / f"{version}.parquet"


def save_features(
    symbol: str,
    features: pd.DataFrame,
    version: str,
    base_path: Path | None = None,
) -> Path:
    """Persist features to a versioned Parquet file and return its path."""
    path = _feature_path(symbol, version, base_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(path)
    return path


def load_features(
    symbol: str,
    version: str,
    base_path: Path | None = None,
) -> pd.DataFrame:
    """Load a feature DataFrame from a versioned Parquet file."""
    path = _feature_path(symbol, version, base_path)
    return pd.read_parquet(path)


def validate_features(features: pd.DataFrame, schema: dict) -> None:
    """Validate a feature DataFrame's metadata using a JSON schema."""
    jsonschema_validate(instance=features.attrs, schema=schema)

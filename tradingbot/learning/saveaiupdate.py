# file: learning/saveaiupdate.py
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

BASE_DIR = Path(__file__).resolve().parents[1] / "state"
MODELS_DIR = BASE_DIR / "models"
REPLAY_DIR = BASE_DIR / "replay_buffers"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPLAY_DIR.mkdir(parents=True, exist_ok=True)

def saveaiupdate(models: Dict[str, Any], replay: Any, meta: Dict[str, Any]) -> Dict[str, str]:
    """Persist models, replay buffer and metadata.

    Returns a mapping of artifact names to file paths.
    """
    paths: Dict[str, str] = {}
    for name, model in models.items():
        path = MODELS_DIR / f"{name}_{uuid4().hex}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        paths[name] = str(path)
    replay_path = REPLAY_DIR / f"replay_{uuid4().hex}.pkl"
    with open(replay_path, "wb") as f:
        pickle.dump(replay, f)
    paths["replay"] = str(replay_path)
    meta_path = BASE_DIR / f"meta_{uuid4().hex}.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    paths["meta"] = str(meta_path)
    return paths

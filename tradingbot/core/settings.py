import os
from typing import Optional, Any

def get_env(name: str, default: Optional[str] = None) -> str:
    return os.environ.get(name, default) if os.environ.get(name) is not None else (default or "")

def get_env_int(name: str, default: int) -> int:
    try:
        return int(get_env(name, str(default)))
    except Exception:
        return default

def get_env_float(name: str, default: float) -> float:
    try:
        return float(get_env(name, str(default)))
    except Exception:
        return default
"""
Training manager for ML & RL with per-asset persistence and auto-resume.

Artifacts:
- Models:
    tradingbot/models/<asset>/ml/checkpoints/model.json
    tradingbot/models/<asset>/rl/ppo/checkpoints/ckpt.json
- Metrics:
    tradingbot/metrics/<asset>/train_ml.jsonl
    tradingbot/metrics/<asset>/train_rl.jsonl
"""
from __future__ import annotations
import os
import json
import pathlib
import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional

try:
    # block training if live enabled
    from tradingbot.core.runtime_api import live_enabled
except Exception:
    def live_enabled(asset: str) -> bool:
        return False

MODELS_DIR = pathlib.Path("tradingbot/models")
METRICS_DIR = pathlib.Path("tradingbot/metrics")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

def _now_iso():
    return datetime.utcnow().isoformat() + "Z"

class _BaseRunner:
    def __init__(self, asset: str, kind: str):
        self.asset = asset
        self.kind = kind  # "ml" or "rl"
        if kind == "ml":
            self.ckpt = MODELS_DIR / asset / "ml" / "checkpoints" / "model.json"
            self.metric_path = METRICS_DIR / asset / "train_ml.jsonl"
        else:
            self.ckpt = MODELS_DIR / asset / "rl" / "ppo" / "checkpoints" / "ckpt.json"
            self.metric_path = METRICS_DIR / asset / "train_rl.jsonl"
        self.ckpt.parent.mkdir(parents=True, exist_ok=True)
        self.metric_path.parent.mkdir(parents=True, exist_ok=True)
        self._thr: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.state: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.ckpt.exists():
            try:
                return json.loads(self.ckpt.read_text(encoding="utf-8"))
            except Exception:
                pass
        # default state
        return {"asset": self.asset, "kind": self.kind, "epoch": 0, "updated_at": _now_iso()}

    def _save(self) -> None:
        self.state["updated_at"] = _now_iso()
        self.ckpt.parent.mkdir(parents=True, exist_ok=True)
        self.ckpt.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def _log_metric(self, row: Dict[str, Any]):
        with self.metric_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def step(self):
        # Minimal placeholder step: increment epoch and write checkpoint.
        # Real trainers should be plugged here (data, model forward/backward, etc).
        self.state["epoch"] = int(self.state.get("epoch", 0)) + 1
        self._save()
        self._log_metric({"ts": _now_iso(), "epoch": self.state["epoch"]})

    def start(self):
        if self._thr and self._thr.is_alive():
            return
        self._stop.clear()
        self._thr = threading.Thread(target=self._run, name=f"trainer-{self.asset}-{self.kind}", daemon=True)
        self._thr.start()

    def _run(self):
        # simple loop; in real impl you'd fetch data, update model, etc.
        while not self._stop.is_set():
            self.step()
            self._stop.wait(2.0)  # 2s per epoch tick

    def stop(self):
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=2.0)

    def status(self) -> Dict[str, Any]:
        return {
            "asset": self.asset,
            "kind": self.kind,
            "running": bool(self._thr and self._thr.is_alive()),
            "epoch": int(self.state.get("epoch", 0)),
            "checkpoint": str(self.ckpt),
            "updated_at": self.state.get("updated_at"),
        }

class _MLRunner(_BaseRunner):
    pass

class _RLRunner(_BaseRunner):
    pass

# registry per asset
_R: Dict[str, Dict[str, _BaseRunner]] = {}

def _get(asset: str, kind: str) -> _BaseRunner:
    asset = asset
    d = _R.setdefault(asset, {})
    if kind not in d:
        d[kind] = _MLRunner(asset) if kind == "ml" else _RLRunner(asset)
    return d[kind]

def start(asset: str, mode: str) -> Dict[str, Any]:
    mode = mode.lower()
    if mode not in ("ml", "rl"):
        return {"ok": False, "reason": "mode must be ml or rl"}
    # Safety: block while live trading is enabled
    try:
        if live_enabled(asset):
            return {"ok": False, "reason": "live is enabled; stop live before training"}
    except Exception:
        pass
    r = _get(asset, mode)
    r.start()
    return {"ok": True, "status": r.status()}

def stop(asset: str, mode: str) -> Dict[str, Any]:
    mode = mode.lower()
    if mode not in ("ml", "rl"):
        return {"ok": False, "reason": "mode must be ml or rl"}
    r = _get(asset, mode)
    r.stop()
    return {"ok": True, "status": r.status()}

def status(asset: str) -> Dict[str, Any]:
    ml = _get(asset, "ml").status()
    rl = _get(asset, "rl").status()
    return {"asset": asset, "ml": ml, "rl": rl}
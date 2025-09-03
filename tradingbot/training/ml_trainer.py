from __future__ import annotations
import json, pathlib, time, math, random
from typing import List, Dict, Any
from tradingbot.training import job_queue as q

MODELS_DIR = pathlib.Path("tradingbot/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def _features_from_prices(prices: List[float]) -> Dict[str, float]:
    if not prices: return {"ema":0,"rsi":50,"mom":0}
    ema = 0.0; k = 2/(14+1)
    for p in prices: ema = p if ema==0 else (p*k + ema*(1-k))
    mom = prices[-1] - prices[max(0,len(prices)-5)]
    # cheap RSI proxy
    ups = [max(0, prices[i]-prices[i-1]) for i in range(1,len(prices))]
    dns = [max(0, prices[i-1]-prices[i]) for i in range(1,len(prices))]
    au = (sum(ups)/len(ups)) if ups else 1e-6
    ad = (sum(dns)/len(dns)) if dns else 1e-6
    rs = au/ad
    rsi = 100 - 100/(1+rs)
    return {"ema":ema, "rsi":rsi, "mom":mom}

def _load_recent_prices(asset: str) -> List[float]:
    # placeholder: integrate your data feed here
    # return last N closes
    return [100 + math.sin(i/10.0)*2 + random.random()*0.5 for i in range(300)]

def _save_model(asset: str, job_id: str, model: dict) -> None:
    d = MODELS_DIR / asset / "ml"
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
    # also save under job id
    (d / f"{job_id}.json").write_text(json.dumps(model, indent=2), encoding="utf-8")

def train(asset: str, job_id: str, params: dict) -> bool:
    prices = _load_recent_prices(asset)
    feats = _features_from_prices(prices)
    # "train" a trivial baseline (store thresholds); replace with sklearn/xgboost
    model = {
        "type":"baseline_thresholds",
        "asset": asset,
        "asof": int(time.time()),
        "params": params,
        "feat_snapshot": feats,
        "rules": {
            "ema_above": feats["ema"] * 1.01,
            "rsi_below": 30,
            "momentum_positive": 0.01
        }
    }
    _save_model(asset, job_id, model)
    return True
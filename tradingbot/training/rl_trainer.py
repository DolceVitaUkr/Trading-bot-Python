from __future__ import annotations
import json, pathlib, random, time, math
from typing import List, Dict, Any
from tradingbot.training import job_queue as q

MODELS_DIR = pathlib.Path("tradingbot/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ACTIONS = [-1, 0, 1]  # short, flat, long

def _load_experience(asset: str, limit: int = 5000) -> List[dict]:
    p = pathlib.Path("tradingbot/state/experience") / f"{asset}.jsonl"
    if not p.exists(): return []
    rows = []
    for ln in p.read_text(encoding="utf-8").splitlines()[-limit:]:
        try: rows.append(json.loads(ln))
        except Exception: pass
    return rows

def _ppo_update(policy: dict, batch: List[dict]) -> dict:
    # toy update: adjust action preferences by mean reward sign
    if not batch: return policy
    mean_r = sum(float(x.get("reward",0.0)) for x in batch)/len(batch)
    pref = policy.get("preferences") or {"-1":0.33,"0":0.34,"1":0.33}
    if mean_r > 0: pref["1"] = min(0.9, pref.get("1",0.33)+0.02)
    else: pref["-1"] = min(0.9, pref.get("-1",0.33)+0.02)
    # normalize
    s = sum(pref.values()) or 1.0
    for k in list(pref.keys()): pref[k] = pref[k]/s
    policy["preferences"] = pref
    return policy

def _save_policy(asset: str, job_id: str, policy: dict) -> None:
    d = MODELS_DIR / asset / "rl"
    d.mkdir(parents=True, exist_ok=True)
    (d / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    (d / f"{job_id}.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")

def train(asset: str, job_id: str, params: dict) -> bool:
    exp = _load_experience(asset, limit=5000)
    # seed policy
    policy = {"type":"ppo-toy", "preferences": {"-1":0.33,"0":0.34,"1":0.33}, "asof": int(time.time())}
    # mini epochs over batches
    bs = 256
    for epoch in range(10):
        if len(exp) < bs: break
        batch = random.sample(exp, bs)
        policy = _ppo_update(policy, batch)
    _save_policy(asset, job_id, policy)
    return True
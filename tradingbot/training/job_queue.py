from __future__ import annotations
import json, pathlib, time, uuid, typing as T

STATE_DIR = pathlib.Path("tradingbot/state")
Q_PATH = STATE_DIR / "training_queue.jsonl"
RUNS_DIR = STATE_DIR / "training_runs"
STATE_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = ("crypto_spot","crypto_futures","forex","options")
MODES = ("ml","rl")

def _append_line(p: pathlib.Path, row: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

def _read_lines(p: pathlib.Path) -> T.List[dict]:
    if not p.exists(): return []
    out = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        try: out.append(json.loads(ln))
        except Exception: pass
    return out

def enqueue(asset: str, mode: str, params: dict|None=None) -> str:
    assert asset in ASSETS, f"invalid asset {asset}"
    assert mode in MODES, f"invalid mode {mode}"
    job_id = str(uuid.uuid4())
    row = {
        "job_id": job_id,
        "asset": asset,
        "mode": mode,
        "params": params or {},
        "status": "queued",
        "tries": 0,
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
    }
    _append_line(Q_PATH, row)
    return job_id

def list_all() -> T.List[dict]:
    return _read_lines(Q_PATH)

def pop_due() -> T.List[dict]:
    """Pop all queued jobs (that aren't already running)."""
    jobs = _read_lines(Q_PATH)
    due, rest = [], []
    for j in jobs:
        if j.get("status") == "queued":
            due.append(j)
        else:
            rest.append(j)
    Q_PATH.write_text("".join(json.dumps(r)+"\n" for r in rest), encoding="utf-8")
    return due

def update_status(job_id: str, status: str, info: dict|None=None) -> None:
    """Update job status (running/completed/failed)."""
    jobs = _read_lines(Q_PATH)
    for j in jobs:
        if j.get("job_id") == job_id:
            j["status"] = status
            j["updated_at"] = int(time.time())
            if info:
                j["info"] = info
    Q_PATH.write_text("".join(json.dumps(j)+"\n" for j in jobs), encoding="utf-8")

def get_status(job_id: str) -> dict|None:
    jobs = _read_lines(Q_PATH)
    for j in jobs:
        if j.get("job_id") == job_id:
            return j
    return None

def save_run(job_id: str, result: dict) -> None:
    """Save training run results."""
    run_path = RUNS_DIR / f"{job_id}.json"
    run_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

def load_run(job_id: str) -> dict|None:
    """Load training run results."""
    run_path = RUNS_DIR / f"{job_id}.json"
    if not run_path.exists():
        return None
    try:
        return json.loads(run_path.read_text(encoding="utf-8"))
    except Exception:
        return None
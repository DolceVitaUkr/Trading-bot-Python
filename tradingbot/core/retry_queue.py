"""
Retry queue & idempotency for order submissions.
Files:
  - tradingbot/state/retry_queue.jsonl
  - tradingbot/state/runtime_dedup.json
"""
from __future__ import annotations
import json, time, pathlib, typing as T

STATE_DIR = pathlib.Path("tradingbot/state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
Q_PATH = STATE_DIR / "retry_queue.jsonl"
DEDUP_PATH = STATE_DIR / "runtime_dedup.json"

def _now() -> int:
    return int(time.time())

def dedup_key(context: dict) -> str:
    return f"{context.get('asset_type')}|{context.get('symbol')}|{context.get('strategy_id')}|{context.get('client_order_id')}"

def is_duplicate(context: dict) -> bool:
    """Check if an order context has already been submitted recently."""
    if not DEDUP_PATH.exists():
        return False
    try:
        m = json.loads(DEDUP_PATH.read_text(encoding="utf-8"))
        now = _now()
        # remove expired
        m = {k: v for k, v in m.items() if v >= now}
        return bool(m.get(dedup_key(context)))
    except Exception:
        return False

def mark_submitted(context: dict, ttl_s: int = 120) -> None:
    """Mark an order as submitted for idempotency within TTL seconds."""
    m = {}
    if DEDUP_PATH.exists():
        try:
            m = json.loads(DEDUP_PATH.read_text(encoding="utf-8"))
        except Exception:
            m = {}
    m[dedup_key(context)] = _now() + int(ttl_s)
    # cleanup old entries
    now = _now()
    m = {k: v for k, v in m.items() if v >= now}
    DEDUP_PATH.write_text(json.dumps(m, indent=2), encoding="utf-8")

def enqueue(context: dict, snapshot_id: str, reason: str, retry_after_s: int = 10, ttl_s: int = 90) -> None:
    """Put a failed order into retry queue."""
    item = {
        "enqueued_at": _now(),
        "not_before": _now() + int(retry_after_s),
        "expires_at": _now() + int(ttl_s),
        "context": context,
        "snapshot_id": snapshot_id,
        "reason": reason,
        "tries": 0,
    }
    with Q_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item) + "\n")

def _iter_queue() -> T.List[dict]:
    rows = []
    if not Q_PATH.exists():
        return rows
    with Q_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows

def depth() -> int:
    """Return current queue depth."""
    return len(_iter_queue())

def drain(retry_cb: T.Callable[[dict], bool], revalidate_cb: T.Callable[[dict], bool]) -> int:
    """Process due items. Returns number processed (accepted or dropped)."""
    rows = _iter_queue()
    if not rows:
        return 0
    now = _now()
    keep, due = [], []
    for r in rows:
        if r.get("expires_at", 0) < now:
            continue
        if r.get("not_before", 0) > now:
            keep.append(r)
        else:
            due.append(r)

    processed, new_keep = 0, []
    for r in due:
        ctx = r.get("context") or {}
        valid = revalidate_cb(ctx) if revalidate_cb else False
        if valid and retry_cb:
            ok = retry_cb(ctx)
        else:
            ok = False
        processed += 1
        if not ok:
            r["tries"] = int(r.get("tries", 0)) + 1
            r["not_before"] = _now() + 5
            new_keep.append(r)

    keep.extend(new_keep)
    with Q_PATH.open("w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r) + "\n")

    return processed

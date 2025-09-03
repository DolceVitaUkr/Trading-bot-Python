import json, time
from pathlib import Path
from typing import Set

class IdempotencyStore:
    def __init__(self, path: Path, ttl_days: int = 7):
        self.path = Path(path)
        self.ttl = ttl_days * 86400
        self._seen: Set[str] = set()
        self._load()
    def _load(self):
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line)
                if time.time() - float(obj.get("ts", 0)) < self.ttl:
                    self._seen.add(obj.get("id"))
            except Exception:
                continue
    def _append(self, cid: str):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": cid, "ts": time.time()}) + "\n")
    def seen(self, cid: str) -> bool:
        return cid in self._seen
    def record(self, cid: str) -> None:
        self._seen.add(cid)
        self._append(cid)
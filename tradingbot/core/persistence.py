import json, os, tempfile, time, shutil
from pathlib import Path
from typing import Any, Dict, Optional

def atomic_write_json(path: Path, data: Any) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)

def safe_read_json(path: Path, default: Any=None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def snapshot_file(path: Path, snapshot_dir: Path, keep: int = 30) -> Optional[Path]:
    path = Path(path); snapshot_dir = Path(snapshot_dir)
    if not path.exists(): return None
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    snap_path = snapshot_dir / f"{path.name}.{ts}.snap"
    shutil.copy2(path, snap_path)
    # trim old
    snaps = sorted(snapshot_dir.glob(f"{path.name}.*.snap"))
    for old in snaps[:-keep]:
        try: old.unlink()
        except: pass
    return snap_path

class WriteAheadLog:
    def __init__(self, wal_path: Path):
        self.wal_path = Path(wal_path)
        self.wal_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.wal_path.exists():
            with open(self.wal_path, "w", encoding="utf-8") as f:
                pass

    def append(self, record: Dict[str, Any]) -> None:
        with open(self.wal_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def read_all(self):
        if not self.wal_path.exists(): return []
        with open(self.wal_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
import json
from pathlib import Path
from typing import Dict, Any, Optional
from .loggerconfig import get_logger

log = get_logger(__name__)

class ProfileManager:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._profiles: Dict[str, Dict[str, Any]] = {}
    def load(self) -> None:
        if not self.path.exists():
            log.warning(f"Profiles file not found: {self.path}")
            self._profiles = {}
            return
        self._profiles = json.loads(self.path.read_text(encoding="utf-8"))
    def get(self, name: str) -> Optional[Dict[str, Any]]:
        return self._profiles.get(name)
    def names(self):
        return list(self._profiles.keys())
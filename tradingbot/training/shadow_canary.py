import time
from typing import Dict, Any

class ShadowCanary:
    def __init__(self, canary_size_pct: float = 0.1):
        self.canary_size_pct = canary_size_pct
        self.active = False
    def start(self):
        self.active = True
    def stop(self):
        self.active = False
    def status(self) -> Dict[str, Any]:
        return {"active": self.active, "size_pct": self.canary_size_pct, "ts": time.time()}
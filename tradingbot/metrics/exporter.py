from typing import Dict, Any

class Metrics:
    def __init__(self):
        self.counters: Dict[str, float] = {}
        self.gauges: Dict[str, float] = {}
    def inc(self, key: str, amt: float = 1.0):
        self.counters[key] = self.counters.get(key, 0.0) + amt
    def set(self, key: str, val: float):
        self.gauges[key] = float(val)
    def as_dict(self) -> Dict[str, Any]:
        return {"counters": dict(self.counters), "gauges": dict(self.gauges)}

metrics = Metrics()
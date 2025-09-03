import asyncio, time
from dataclasses import dataclass
from typing import Dict, Tuple
from .settings import get_env_float, get_env_int

@dataclass
class TokenBucket:
    rate_per_sec: float
    capacity: int
    tokens: float = 0.0
    last_ts: float = 0.0
    def __post_init__(self):
        self.tokens = float(self.capacity)
        self.last_ts = time.monotonic()
    async def acquire(self, n: int = 1):
        while True:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now - self.last_ts) * self.rate_per_sec)
            self.last_ts = now
            if self.tokens >= n:
                self.tokens -= n
                return
            await asyncio.sleep( max(0.0, (n - self.tokens) / max(self.rate_per_sec, 1e-9)) )

class RateLimitManager:
    def __init__(self):
        headroom = get_env_float("RATE_LIMIT_PCT", 0.9)
        # Defaults (can be overridden by env)
        bybit_read_rps  = get_env_int("BYBIT_READ_RPS", 8)
        bybit_trade_rps = get_env_int("BYBIT_TRADE_RPS", 2)
        ibkr_read_rps   = get_env_int("IBKR_READ_RPS", 3)
        ibkr_trade_rps  = get_env_int("IBKR_TRADE_RPS", 1)
        self.buckets: Dict[Tuple[str,str], TokenBucket] = {
            ("bybit","read"):  TokenBucket(bybit_read_rps*headroom, max(1,int(bybit_read_rps*headroom))),
            ("bybit","trade"): TokenBucket(bybit_trade_rps*headroom, max(1,int(bybit_trade_rps*headroom))),
            ("ibkr","read"):   TokenBucket(ibkr_read_rps*headroom, max(1,int(ibkr_read_rps*headroom))),
            ("ibkr","trade"):  TokenBucket(ibkr_trade_rps*headroom, max(1,int(ibkr_trade_rps*headroom))),
        }
    def bucket(self, venue: str, kind: str) -> TokenBucket:
        return self.buckets[(venue, kind)]

# Global manager (import and use)
_rlm = RateLimitManager()
def get_bucket(venue: str, kind: str) -> TokenBucket:
    return _rlm.bucket(venue, kind)
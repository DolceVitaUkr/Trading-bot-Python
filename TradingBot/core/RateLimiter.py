import asyncio
import time
from collections import deque
from functools import wraps
import random
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections import deque

log = logging.getLogger(__name__)

class RateLimiter:
    """
    A flexible rate limiter that supports token bucket for RPS limits
    and a sliding window for historical data pacing.
    """

    def __init__(self, rps_limit: int, historical_limit: int, historical_period: int):
        # Token bucket settings for general requests
        self.rps_limit = rps_limit
        self.tokens = float(rps_limit)
        self.last_refill_time = time.monotonic()

        # Sliding window for historical data pacing
        self.historical_limit = historical_limit  # e.g., 60 requests
        self.historical_period = historical_period  # e.g., 600 seconds (10 minutes)
        self.historical_requests: "deque[float]" = deque()

    async def _refill_tokens(self):
        """Refills tokens based on the time elapsed since the last refill."""
        now = time.monotonic()
        time_passed = now - self.last_refill_time
        new_tokens = time_passed * self.rps_limit
        if new_tokens > 0:
            self.tokens = min(self.rps_limit, self.tokens + new_tokens)
            self.last_refill_time = now

    async def wait_for_token(self):
        """Waits until a token is available for a general request."""
        await self._refill_tokens()
        while self.tokens < 1:
            await asyncio.sleep(0.05)
            await self._refill_tokens()
        self.tokens -= 1

    async def wait_for_historical_slot(self):
        """Waits until a slot is available for a historical data request."""
        await self.wait_for_token() # First, ensure we respect the general RPS limit

        now = time.monotonic()

        # Remove old requests from the queue that are outside the sliding window
        while self.historical_requests and self.historical_requests[0] <= now - self.historical_period:
            self.historical_requests.popleft()

        if len(self.historical_requests) >= self.historical_limit:
            # Calculate how long to wait for the oldest request to expire
            time_to_wait = (self.historical_requests[0] + self.historical_period) - now
            jitter = random.uniform(0.1, 0.5) # Add jitter to avoid thundering herd
            wait_time = time_to_wait + jitter

            log.warning(f"Historical data limit reached ({self.historical_limit} reqs / {self.historical_period}s). "
                        f"Waiting for {wait_time:.2f} seconds.")
            await asyncio.sleep(wait_time)

        self.historical_requests.append(time.monotonic())

    def limit(self, f):
        """Decorator to apply the general RPS rate limit to a function."""
        @wraps(f)
        async def decorated_function(*args, **kwargs):
            await self.wait_for_token()
            return await f(*args, **kwargs)
        return decorated_function

# A global instance can be created and imported where needed.
# IBKR limits: ~50 req/sec for Web API, 60 req/10 min for historical data.
# We'll use a slightly lower limit to be safe.
ibkr_RateLimiter = RateLimiter(rps_limit=40, historical_limit=58, historical_period=600)

# Bybit limits: ~50 req/sec for most endpoints, but we'll be conservative.
bybit_RateLimiter = RateLimiter(rps_limit=45, historical_limit=50, historical_period=60)

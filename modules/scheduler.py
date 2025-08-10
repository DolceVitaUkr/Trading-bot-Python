# modules/scheduler.py

import asyncio
import logging
from typing import Callable, Awaitable, Dict, Optional
from datetime import datetime, timezone


logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


class _Job:
    def __init__(self, name: str, coro_factory: Callable[[], Awaitable[None]], interval_seconds: float):
        self.name = name
        self.coro_factory = coro_factory
        self.interval_seconds = float(interval_seconds)
        self._task: Optional[asyncio.Task] = None
        self._cancelled = False

    async def _runner(self):
        try:
            while not self._cancelled:
                started = datetime.now(timezone.utc)
                try:
                    await self.coro_factory()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.warning(f"[scheduler] job '{self.name}' failed: {e}")
                # Maintain steady cadence (drift-minimized)
                elapsed = (datetime.now(timezone.utc) - started).total_seconds()
                delay = max(0.0, self.interval_seconds - elapsed)
                await asyncio.sleep(delay)
        except asyncio.CancelledError:
            pass

    def start(self):
        if self._task and not self._task.done():
            return
        self._cancelled = False
        self._task = asyncio.create_task(self._runner(), name=f"scheduler:{self.name}")

    def cancel(self):
        self._cancelled = True
        if self._task and not self._task.done():
            self._task.cancel()


class LightweightScheduler:
    """
    Minimal asyncio-based scheduler:
      - add_job(name, coro, interval_seconds)
      - cancel_job(name)
      - start(), stop()
    No external dependencies.
    """

    def __init__(self):
        self._jobs: Dict[str, _Job] = {}
        self._started = False

    def add_job(self, name: str, coro: Callable[[], Awaitable[None]], interval_seconds: float) -> None:
        if name in self._jobs:
            # replace in-place
            self.cancel_job(name)
        job = _Job(name=name, coro_factory=coro, interval_seconds=interval_seconds)
        self._jobs[name] = job
        if self._started:
            job.start()
        logger.info(f"[scheduler] added job '{name}' every {interval_seconds}s")

    def cancel_job(self, name: str) -> None:
        job = self._jobs.pop(name, None)
        if job:
            job.cancel()
            logger.info(f"[scheduler] cancelled job '{name}'")

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        for job in self._jobs.values():
            job.start()
        logger.info("[scheduler] started")

    def stop(self) -> None:
        if not self._started:
            return
        for name in list(self._jobs.keys()):
            self.cancel_job(name)
        self._started = False
        logger.info("[scheduler] stopped")

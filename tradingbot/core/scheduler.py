import asyncio, time
from pathlib import Path
from typing import Callable, Awaitable
from .loggerconfig import get_logger
log = get_logger(__name__)

class Scheduler:
    def __init__(self):
        self._tasks = []
    def every(self, seconds: int, coro_fn: Callable[[], Awaitable[None]]):
        async def _runner():
            while True:
                try:
                    await coro_fn()
                except Exception as e:
                    log.error(f"Scheduled task failed: {e}")
                await asyncio.sleep(seconds)
        self._tasks.append(asyncio.create_task(_runner()))
    async def start(self):
        await asyncio.gather(*self._tasks)

# NEW: helper to register catalog daily refresh
async def register_catalog_refresh(sched: Scheduler, catalog, bybit_adapter=None, ibkr_adapter=None, symbols=None):
    async def _refresh():
        try:
            await catalog.refresh_from_venues(bybit_adapter, ibkr_adapter, symbols)
        except Exception as e:
            log.error(f"catalog refresh failed: {e}")
    sched.every(24*3600, _refresh)

# NEW: helper to register periodic reconciler
async def register_reconciler(sched: Scheduler, bybit_adapter=None, ibkr_adapter=None, interval_sec: int = 60):
    from .reconciler import reconcile_once
    async def _run():
        try:
            await reconcile_once(bybit_adapter, ibkr_adapter)
        except Exception as e:
            log.error(f"reconciler tick failed: {e}")
    sched.every(interval_sec, _run)
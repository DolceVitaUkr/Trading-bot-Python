# scheduler.py

import time
import threading
from dataclasses import dataclass
from typing import Callable, Optional, Dict


@dataclass
class _Job:
    name: str
    func: Callable[[], None]
    interval: float  # seconds
    next_run: float


class JobScheduler:
    """
    Lightweight interval scheduler.
    - all jobs run in the same thread (this module starts that thread in main.py)
    - each job executed sequentially; keep funcs short
    """

    def __init__(self):
        self._jobs: Dict[str, _Job] = {}
        self._lock = threading.Lock()
        self._running = False

    def every(self, *, seconds: Optional[int] = None, minutes: Optional[int] = None,
              name: str, func: Callable[[], None]) -> None:
        if seconds is None and minutes is None:
            raise ValueError("Provide seconds or minutes")
        interval = float(seconds if seconds is not None else minutes * 60)
        now = time.time()
        with self._lock:
            self._jobs[name] = _Job(name=name, func=func, interval=interval, next_run=now + interval)

    def cancel(self, name: str) -> None:
        with self._lock:
            self._jobs.pop(name, None)

    def run_pending(self) -> None:
        now = time.time()
        due: list[_Job] = []
        with self._lock:
            for j in self._jobs.values():
                if now >= j.next_run:
                    due.append(j)
        for j in due:
            try:
                j.func()
            finally:
                with self._lock:
                    j.next_run = time.time() + j.interval

    def run_forever(self, tick: float = 1.0) -> None:
        self._running = True
        while self._running:
            try:
                self.run_pending()
            except Exception:
                # swallow; individual jobs should log
                pass
            time.sleep(tick)

    def stop(self) -> None:
        self._running = False

# modules/health_monitor.py

import threading
import time
import logging
from typing import Callable, Optional, Dict, Any

from modules.telegram_bot import TelegramNotifier

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


class HealthMonitor:
    """
    Lightweight liveness & recap pings to Telegram.

    Features
    --------
    - Heartbeat: sends a short "I'm alive" message at a configurable cadence.
    - Watchdog: alerts if no heartbeat recorded for `watchdog_timeout_sec`.
    - Recap: calls a user-provided status provider to render a compact summary
             (balances, open trades, last actions) at a configurable cadence.

    Typical wiring
    --------------
        hm = HealthMonitor(notifier, mode="paper",
                           heartbeat_interval_sec=900,
                           recap_interval_sec=3600,
                           watchdog_timeout_sec=1800)
        hm.set_status_provider(bot_status_fn)  # returns dict or str
        hm.start()

        # ... in your main loop:
        hm.record_heartbeat("live_loop")  # every loop tick

        # on shutdown:
        hm.stop()

    Notes
    -----
    - All timers are best-effort, jitter-free simple threads.
    - No hard dependency on your bot/exchange. Pass a status provider to enrich recaps.
    """

    def __init__(
        self,
        notifier: TelegramNotifier,
        mode: str = "paper",  # "paper" | "live"
        *,
        heartbeat_interval_sec: int = 15 * 60,
        recap_interval_sec: Optional[int] = None,
        watchdog_timeout_sec: int = 30 * 60,
        quiet_heartbeat: bool = True,
    ):
        self.notifier = notifier
        self.mode = mode
        self.heartbeat_interval_sec = int(heartbeat_interval_sec)
        self.recap_interval_sec = int(recap_interval_sec) if recap_interval_sec else None
        self.watchdog_timeout_sec = int(watchdog_timeout_sec)
        self.quiet_heartbeat = bool(quiet_heartbeat)

        self._status_provider: Optional[Callable[[], Any]] = None
        self._stop_event = threading.Event()
        self._last_heartbeat_ts = 0.0

        self._t_heartbeat: Optional[threading.Thread] = None
        self._t_watchdog: Optional[threading.Thread] = None
        self._t_recap: Optional[threading.Thread] = None

    # ---------------- Public API ---------------- #

    def set_status_provider(self, provider: Callable[[], Any]) -> None:
        """
        provider() -> dict | str
        If dict, keys used: balance, equity, open_positions (int), last_trade (str), mode (str), symbol (str)
        """
        self._status_provider = provider

    def record_heartbeat(self, source: str = "unknown") -> None:
        """Mark progress from your trading loop or any worker."""
        self._last_heartbeat_ts = time.time()
        logger.debug(f"[HealthMonitor] heartbeat from {source}")

    def start(self) -> None:
        if self._t_heartbeat or self._t_watchdog:
            return
        self._stop_event.clear()
        self._t_heartbeat = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._t_watchdog = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._t_heartbeat.start()
        self._t_watchdog.start()
        if self.recap_interval_sec:
            self._t_recap = threading.Thread(target=self._recap_loop, daemon=True)
            self._t_recap.start()
        logger.info("HealthMonitor started")

    def stop(self) -> None:
        self._stop_event.set()
        for t in (self._t_heartbeat, self._t_watchdog, self._t_recap):
            if t and t.is_alive():
                t.join(timeout=1.5)
        self._t_heartbeat = self._t_watchdog = self._t_recap = None
        logger.info("HealthMonitor stopped")

    # ---------------- Internals ---------------- #

    def _heartbeat_loop(self):
        # Send a quiet ping at interval. Include mode + short status if available.
        # Also initialize last heartbeat so watchdog doesn't scream on boot.
        self._last_heartbeat_ts = time.time()
        while not self._stop_event.wait(self.heartbeat_interval_sec):
            try:
                text = self._render_heartbeat()
                if not self.quiet_heartbeat or self.mode == "live":
                    self.notifier.send_message_sync(text, format="log")
                else:
                    # Quiet mode: still log locally
                    logger.info(text.replace("<b>", "").replace("</b>", ""))
            except Exception as e:
                logger.warning(f"Heartbeat send failed: {e}")

    def _watchdog_loop(self):
        # If no heartbeat has been recorded for too long, alert.
        poll = max(5, min(30, self.watchdog_timeout_sec // 6))
        while not self._stop_event.wait(poll):
            try:
                last = self._last_heartbeat_ts
                if not last:
                    continue
                silence = time.time() - last
                if silence >= self.watchdog_timeout_sec:
                    self.notifier.send_message_sync(
                        {"type": "ALERT", "message": f"⚠️ No heartbeat in {int(silence)}s (mode={self.mode}). "
                                                     f'Bot may be stalled/offline.'},
                        format="alert"
                    )
                    # After alert, push the threshold forward to avoid spamming
                    self._last_heartbeat_ts = time.time()
            except Exception as e:
                logger.warning(f"Watchdog check failed: {e}")

    def _recap_loop(self):
        while not self._stop_event.wait(self.recap_interval_sec):
            try:
                text = self._render_recap()
                self.notifier.send_message_sync(text, format="text")
            except Exception as e:
                logger.warning(f"Recap send failed: {e}")

    def _render_heartbeat(self) -> str:
        base = f"<b>Heartbeat</b> | mode={self.mode}"
        if self._status_provider:
            try:
                st = self._status_provider()
                if isinstance(st, dict):
                    b = st.get("balance")
                    e = st.get("equity")
                    sym = st.get("symbol")
                    extras = []
                    if sym:
                        extras.append(f"sym={sym}")
                    if b is not None:
                        extras.append(f"bal=${float(b):,.2f}")
                    if e is not None:
                        extras.append(f"eq=${float(e):,.2f}")
                    if extras:
                        base += " | " + " ".join(extras)
                elif isinstance(st, str) and st.strip():
                    base += f" | {st.strip()}"
            except Exception:
                pass
        return base

    def _render_recap(self) -> str:
        if not self._status_provider:
            return f"🧾 <b>{self.mode.upper()} Recap</b>\n(no status provider wired)"
        st = self._status_provider()
        if isinstance(st, dict):
            lines = [
                f"🧾 <b>{self.mode.upper()} Recap</b>",
                f"Balance: ${float(st.get('balance', 0.0)) :,.2f}",
                f"Equity:  ${float(st.get('equity', 0.0))  :,.2f}",
                f"Open positions: {int(st.get('open_positions', 0))}",
            ]
            lt = st.get("last_trade")
            if lt:
                lines.append(f"Last trade: {lt}")
            return "\n".join(lines)
        elif isinstance(st, str):
            return f"🧾 <b>{self.mode.upper()} Recap</b>\n{st.strip()}"
        return f"🧾 <b>{self.mode.upper()} Recap</b>\n(status unavailable)"

# modules/notification_manager.py

import time
import logging
from typing import Dict, Any, Optional
from modules.telegram_bot import TelegramNotifier

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


class NotificationManager:
    """
    Policy-driven wrapper around TelegramNotifier.
    - Paper: batched recap every N minutes (configurable)
    - Live: 'quiet' / 'normal' / 'verbose' throttle for trade/status
    - Heartbeat: lightweight "I'm alive" ping every M minutes (optional)
    """

    def __init__(
        self,
        notifier: Optional[TelegramNotifier] = None,
        paper_recap_minutes: int = 60,
        live_alert_level: str = "normal",   # quiet|normal|verbose
        heartbeat_minutes: int = 30,
    ):
        self.notifier = notifier or TelegramNotifier(disable_async=False)
        self.paper_recap_minutes = paper_recap_minutes
        self.live_alert_level = live_alert_level
        self.heartbeat_minutes = heartbeat_minutes

        # rolling buffers
        self._paper_events: list[Dict[str, Any]] = []
        self._last_paper_flush = 0.0
        self._last_heartbeat = 0.0

    # ----- external API (called by bot/executor) -----

    def apply_prefs(self, prefs: Dict[str, Any]):
        self.paper_recap_minutes = int(prefs.get("paper_recap_minutes", self.paper_recap_minutes))
        self.live_alert_level = str(prefs.get("live_alert_level", self.live_alert_level)).lower()

    def notify_trade(self, event: Dict[str, Any]):
        """
        event: {symbol, side, qty, price, status, opened, closed, pnl, return_pct, leverage, meta:{mode:'paper'|'live', ...}}
        """
        mode = (event.get("meta", {}) or {}).get("mode", "live")
        if mode == "paper":
            self._paper_events.append(event)
            self._maybe_flush_paper()
            return

        # live
        lvl = self.live_alert_level
        if lvl == "quiet":
            # Only closes or errors
            if event.get("closed") or (event.get("status") in ("closed", "closed_partial")):
                self._send(self._fmt_trade(event))
        elif lvl == "normal":
            # Opens + closes
            if event.get("opened") or event.get("closed"):
                self._send(self._fmt_trade(event))
        else:  # verbose
            self._send(self._fmt_trade(event))

    def notify_status(self, text: str, importance: str = "info"):
        """live status messages (throttled by level if you want)"""
        if self.live_alert_level == "quiet" and importance == "info":
            return
        self._send(text)

    def notify_error(self, text: str):
        self._send({"type": "ALERT", "message": text}, fmt="alert")

    def heartbeat(self):
        """Call periodically; sends 'alive' pings at configured cadence."""
        now = time.time()
        if self.heartbeat_minutes <= 0:
            return
        if now - self._last_heartbeat >= self.heartbeat_minutes * 60:
            self._last_heartbeat = now
            self._send("💓 Bot heartbeat: alive and monitoring.")

    # ----- internals -----

    def _maybe_flush_paper(self):
        now = time.time()
        if not self._paper_events:
            return
        if self.paper_recap_minutes <= 0:
            # immediate send (not typical)
            self._send(self._fmt_paper_recap(self._paper_events))
            self._paper_events.clear()
            self._last_paper_flush = now
            return

        if now - self._last_paper_flush >= self.paper_recap_minutes * 60:
            self._send(self._fmt_paper_recap(self._paper_events))
            self._paper_events.clear()
            self._last_paper_flush = now

    def _fmt_paper_recap(self, events: list[Dict[str, Any]]) -> str:
        n = len(events)
        realized = [e for e in events if e.get("closed")]
        p_count = len(realized)
        pnl_sum = sum(float(e.get("pnl") or 0.0) for e in realized)
        lines = [f"📝 Paper Recap ({n} events, {p_count} closed)"]
        for e in realized[-10:]:  # last 10 closed
            lines.append(self._line_trade(e))
        lines.append(f"\nTotal realized PnL: {pnl_sum:.2f} USDT")
        return "\n".join(lines)

    def _fmt_trade(self, e: Dict[str, Any]) -> str:
        title = "TRADE CLOSE" if e.get("closed") else "TRADE OPEN"
        return f"📊 <b>{title}</b>\n" + self._line_trade(e)

    def _line_trade(self, e: Dict[str, Any]) -> str:
        side = str(e.get("side","")).upper()
        sym = e.get("symbol")
        qty = e.get("qty")
        px  = e.get("price")
        st  = e.get("status","")
        pnl = e.get("pnl")
        rp  = e.get("return_pct")
        parts = [
            f"Pair: <code>{sym}</code>",
            f"Side: <b>{side}</b>",
            f"Qty: <code>{qty}</code>",
            f"Px: <code>{px}</code>",
            f"Status: {st}",
        ]
        if pnl is not None:
            parts.append(f"PnL: <b>{float(pnl):.2f}</b>")
        if rp is not None:
            parts.append(f"Return: <b>{float(rp):.2f}%</b>")
        return "\n".join(parts)

    def _send(self, content, fmt: str = "text"):
        try:
            self.notifier.send_message(content, format=fmt)
        except Exception:
            logger.exception("Notification send failed")

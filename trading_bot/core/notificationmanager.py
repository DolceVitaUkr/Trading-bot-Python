import time
import logging
from typing import Dict, Any, Optional, List
from trading_bot.core.telegram_bot import TelegramNotifier

logger = logging.getLogger(__name__)


class NotificationManager:
    """
    Policy-driven wrapper around TelegramNotifier.
    - Paper: batched recap every N minutes (configurable)
    - Live: 'quiet' / 'normal' / 'verbose' throttle for trade/status
    - Heartbeat: lightweight "I'm alive" ping every M minutes
    """

    def __init__(
        self,
        notifier: Optional[TelegramNotifier] = None,
        paper_recap_minutes: int = 60,
        live_alert_level: str = "normal",  # quiet|normal|verbose
        heartbeat_minutes: int = 30,
    ):
        """
        Initializes the NotificationManager.
        """
        self.notifier = notifier or TelegramNotifier(disable_async=False)
        self.paper_recap_minutes = int(paper_recap_minutes)
        self.live_alert_level = str(live_alert_level).lower()
        self.heartbeat_minutes = int(heartbeat_minutes)

        # rolling buffers
        self._paper_events: List[Dict[str, Any]] = []
        self._last_paper_flush = 0.0
        self._last_heartbeat = 0.0

        # lightweight snapshot for recaps/heartbeat context
        self._snapshot: Dict[str, Any] = {}

    # ----- external API (called by bot/executor) -----

    def apply_prefs(self, prefs: Dict[str, Any]):
        """
        Applies notification preferences.
        """
        self.paper_recap_minutes = int(
            prefs.get("paper_recap_minutes", self.paper_recap_minutes))
        self.live_alert_level = str(
            prefs.get("live_alert_level", self.live_alert_level)).lower()

    def update_metrics_snapshot(self, **kwargs):
        """
        Store latest metrics to show in recaps.
        """
        self._snapshot.update(
            {k: v for k, v in kwargs.items() if v is not None})

    async def notify_trade(self, event: Dict[str, Any]):
        """
        Notifies of a trade event.
        """
        mode = ((event.get("meta") or {}).get("mode") or "live").lower()
        if mode == "paper":
            self._paper_events.append(event)
            return  # flush happens in tick()
        # live
        lvl = self.live_alert_level
        if lvl == "quiet":
            if event.get("closed") or (
                    event.get("status") in ("closed", "closed_partial")):
                await self._send(self._fmt_trade(event))
        elif lvl == "normal":
            if event.get("opened") or event.get("closed"):
                await self._send(self._fmt_trade(event))
        else:  # verbose
            await self._send(self._fmt_trade(event))

    async def notify_status(self, text: str, importance: str = "info"):
        """
        Notifies of a status event.
        """
        if self.live_alert_level == "quiet" and importance == "info":
            return
        await self._send(text)

    async def notify_error(self, text: str):
        """
        Notifies of an error event.
        """
        await self._send({"type": "ALERT", "message": text}, fmt="alert")

    async def tick(self):
        """
        Call this periodically from your main loop.
        """
        await self._heartbeat_maybe()
        await self._maybe_flush_paper()

    # ----- internals -----

    async def _heartbeat_maybe(self):
        now = time.time()
        if self.heartbeat_minutes <= 0:
            return
        if now - self._last_heartbeat >= self.heartbeat_minutes * 60:
            self._last_heartbeat = now
            snap = self._snapshot
            extra = []
            if "symbol" in snap and "price" in snap:
                extra.append(f"{snap['symbol']}={snap['price']}")
            if "equity" in snap:
                extra.append(f"equity={snap['equity']:.2f}")
            if "balance" in snap:
                extra.append(f"balance={snap['balance']:.2f}")
            suffix = " | ".join(extra) if extra else ""
            await self._send(f"💓 Bot heartbeat: alive. {suffix}")

    async def _maybe_flush_paper(self):
        now = time.time()
        if not self._paper_events:
            return
        if self.paper_recap_minutes <= 0:
            await self._send(self._fmt_paper_recap(self._paper_events))
            self._paper_events.clear()
            self._last_paper_flush = now
            return
        if now - self._last_paper_flush >= self.paper_recap_minutes * 60:
            await self._send(self._fmt_paper_recap(self._paper_events))
            self._paper_events.clear()
            self._last_paper_flush = now

    def _fmt_paper_recap(self, events: List[Dict[str, Any]]) -> str:
        n = len(events)
        realized = [e for e in events if e.get("closed")]
        p_count = len(realized)
        pnl_sum = sum(float(e.get("pnl") or 0.0) for e in realized)
        snap = self._snapshot
        head = f"📝 Paper Recap ({n} events, {p_count} closed)"
        if snap:
            head += (f" | equity={snap.get('equity', '?')} "
                     f"balance={snap.get('balance', '?')}")
        lines = [head]
        for e in realized[-10:]:
            lines.append(self._line_trade(e))
        lines.append(f"\nTotal realized PnL: {pnl_sum:.2f} USDT")
        return "\n".join(lines)

    def _fmt_trade(self, e: Dict[str, Any]) -> str:
        title = "TRADE CLOSE" if e.get("closed") else "TRADE OPEN"
        return f"📊 <b>{title}</b>\n" + self._line_trade(e)

    def _line_trade(self, e: Dict[str, Any]) -> str:
        side = str(e.get("side", "")).upper()
        sym = e.get("symbol")
        qty = e.get("qty")
        px = e.get("price")
        st = e.get("status", "")
        pnl = e.get("pnl")
        rp = e.get("return_pct")
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

    async def _send(self, content, fmt: str = "text"):
        try:
            await self.notifier.send_message_async(content, format=fmt)
        except Exception:
            logger.exception("Notification send failed")

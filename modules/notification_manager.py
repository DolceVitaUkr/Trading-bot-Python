# modules/notification_manager.py

import time
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from modules.telegram_bot import TelegramNotifier

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


@dataclass
class TradeEvent:
    symbol: str
    side: str
    qty: float
    price: float
    pnl: Optional[float] = None
    return_pct: Optional[float] = None
    leverage: Optional[float] = None
    opened: bool = False
    closed: bool = False
    status: str = "open"
    meta: Dict[str, Any] = field(default_factory=dict)


class NotificationManager:
    """
    Policy-driven wrapper around TelegramNotifier.

    - Paper: batched recap every N minutes
    - Live: 'quiet' / 'normal' / 'verbose' throttle for trade/status
    - Heartbeat: "I'm alive" ping every M minutes
    - Metrics snapshot: lets heartbeat/digest include balances/price
    """

    def __init__(
        self,
        notifier: Optional[TelegramNotifier] = None,
        *,
        # keep both names to handle older callers
        paper_recap_minutes: Optional[int] = None,
        paper_recap_min: Optional[int] = None,
        live_alert_level: str = "normal",          # quiet|normal|verbose
        heartbeat_minutes: Optional[int] = None,
        heartbeat_min: Optional[int] = None,
        mode: str = "live",                        # "paper" or "live" (used only as default when meta missing)
    ):
        self.notifier = notifier or TelegramNotifier(disable_async=False)
        self.live_alert_level = str(live_alert_level).lower()
        self.paper_recap_minutes = int(paper_recap_minutes if paper_recap_minutes is not None else (paper_recap_min or 60))
        self.heartbeat_minutes = int(heartbeat_minutes if heartbeat_minutes is not None else (heartbeat_min or 30))
        self.default_mode = mode

        # rolling buffers
        self._paper_events: List[TradeEvent] = []
        self._last_paper_flush = 0.0
        self._last_heartbeat = 0.0

        # metrics snapshot (optional)
        self._metrics: Dict[str, Any] = {}

    # ---------- prefs & snapshots ----------

    def apply_prefs(self, prefs: Dict[str, Any]):
        if "paper_recap_minutes" in prefs:
            self.paper_recap_minutes = int(prefs["paper_recap_minutes"])
        if "live_alert_level" in prefs:
            self.live_alert_level = str(prefs["live_alert_level"]).lower()

    def update_metrics_snapshot(self, **kwargs):
        """
        Accepts arbitrary metrics like:
          price=..., symbol=..., equity=..., balance=...
        """
        self._metrics.update(kwargs)

    # ---------- main API ----------

    def notify_trade(self, event: Any):
        e = event.__dict__ if hasattr(event, "__dict__") else event
        mode = (e.get("meta", {}) or {}).get("mode", "live")

        if mode == "paper":
            self._paper_events.append(e)
            self._maybe_flush_paper()
            return

        # live throttling
        lvl = self.live_alert_level
        if lvl == "quiet":
            if e.closed or (e.status in ("closed", "closed_partial")):
                self._send(self._fmt_trade(e), fmt="text")
        elif lvl == "normal":
            if e.opened or e.closed:
                self._send(self._fmt_trade(e), fmt="text")
        else:  # verbose
            self._send(self._fmt_trade(e), fmt="text")

    def notify_status(self, text: str, importance: str = "info"):
        if self.live_alert_level == "quiet" and importance == "info":
            return
        self._send(text, fmt="text")

    def notify_error(self, text: str):
        self._send({"type": "ALERT", "message": text}, fmt="alert")

    def tick(self):
        """
        Call this once per main loop tick to drive heartbeat + recap schedule.
        """
        self._heartbeat()
        self._maybe_flush_paper()

    # ---------- internals ----------

    def _normalize_event(self, e: "TradeEvent | Dict[str, Any]") -> TradeEvent:
        if isinstance(e, TradeEvent):
            return e
        return TradeEvent(
            symbol=e.get("symbol", ""),
            side=e.get("side", ""),
            qty=float(e.get("qty", 0)),
            price=float(e.get("price", 0)),
            status=e.get("status"),
            opened=bool(e.get("opened", False)),
            closed=bool(e.get("closed", False)),
            pnl=e.get("pnl"),
            return_pct=e.get("return_pct"),
            leverage=e.get("leverage"),
            meta=e.get("meta", {}),
        )

    def _heartbeat(self):
        now = time.time()
        if self.heartbeat_minutes <= 0:
            return
        if now - self._last_heartbeat >= self.heartbeat_minutes * 60:
            self._last_heartbeat = now
            parts = ["💓 Bot heartbeat: alive."]
            if self._metrics:
                bal = self._metrics.get("balance")
                eq = self._metrics.get("equity")
                sym = self._metrics.get("symbol")
                px = self._metrics.get("price")
                if bal is not None:
                    parts.append(f"Balance: {bal:,.2f}")
                if eq is not None:
                    parts.append(f"Equity: {eq:,.2f}")
                if sym and px is not None:
                    parts.append(f"{sym} @ {px}")
            self._send("\n".join(parts), fmt="text")

    def _maybe_flush_paper(self):
        now = time.time()
        if not self._paper_events:
            return
        if self.paper_recap_minutes <= 0:
            self._send(self._fmt_paper_recap(self._paper_events), fmt="text")
            self._paper_events.clear()
            self._last_paper_flush = now
            return
        if now - self._last_paper_flush >= self.paper_recap_minutes * 60:
            self._send(self._fmt_paper_recap(self._paper_events), fmt="text")
            self._paper_events.clear()
            self._last_paper_flush = now

    def _fmt_paper_recap(self, events: List[TradeEvent]) -> str:
        n = len(events)
        realized = [e for e in events if e.closed or (e.status in ("closed", "closed_partial"))]
        p_count = len(realized)
        pnl_sum = sum(float(e.pnl or 0.0) for e in realized)
        lines = [f"📝 Paper Recap ({n} events, {p_count} closed)"]
        for e in realized[-10:]:
            lines.append(self._line_trade(e))
        lines.append(f"\nTotal realized PnL: {pnl_sum:.2f} USDT")
        return "\n".join(lines)

    def _fmt_trade(self, e: TradeEvent) -> str:
        title = "TRADE CLOSE" if (e.closed or (e.status in ("closed", "closed_partial"))) else "TRADE OPEN"
        return f"📊 <b>{title}</b>\n" + self._line_trade(e)

    def _line_trade(self, e: TradeEvent) -> str:
        side = str(e.side).upper()
        parts = [
            f"Pair: <code>{e.symbol}</code>",
            f"Side: <b>{side}</b>",
            f"Qty: <code>{e.qty}</code>",
            f"Px: <code>{e.price}</code>",
        ]
        if e.status:
            parts.append(f"Status: {e.status}")
        if e.pnl is not None:
            parts.append(f"PnL: <b>{float(e.pnl):.2f}</b>")
        if e.return_pct is not None:
            parts.append(f"Return: <b>{float(e.return_pct):.2f}%</b>")
        if e.leverage:
            parts.append(f"Lev: {e.leverage}x")
        return "\n".join(parts)

    def _send(self, content, fmt: str = "text"):
        try:
            self.notifier.send_message(content, format=fmt)
        except Exception:
            logger.exception("Notification send failed")

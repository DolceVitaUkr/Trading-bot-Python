#modules/notification_manager.py
import time
import math
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal

import config
from modules.telegram_bot import TelegramNotifier

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

AlertLevel = Literal["quiet", "normal", "verbose"]
Mode = Literal["paper", "live"]


@dataclass
class TradeEvent:
    symbol: str
    side: str
    qty: float
    price: float
    pnl: Optional[float] = None
    return_pct: Optional[float] = None
    leverage: Optional[float] = None
    opened: Optional[str] = None       # ISO str
    closed: Optional[str] = None       # ISO str
    status: Optional[str] = None       # open/closed/simulated/...
    meta: Dict = field(default_factory=dict)


class NotificationManager:
    """
    Centralized message policy:
      - Paper mode: buffer trade events and send a digest every N minutes.
      - Live mode: send messages in real-time with adjustable verbosity.
      - Heartbeat pings so you know the bot is alive even if idle.

    Usage:
      nm = NotificationManager(notifier, mode="paper")
      nm.notify_trade({...})  # call whenever a trade happens
      nm.notify_status("Starting scan...")
      nm.notify_error("Bybit down?")
      loop:
         nm.tick()  # call once per main loop iteration (cheap)
    """

    def __init__(
        self,
        notifier: Optional[TelegramNotifier] = None,
        *,
        mode: Mode = "paper",
        paper_recap_min: Optional[int] = None,
        live_alert_level: Optional[AlertLevel] = None,
        heartbeat_min: Optional[int] = None,
    ):
        self.notifier = notifier or TelegramNotifier(disable_async=True)

        # prefs (with config fallbacks)
        self.mode: Mode = mode
        self.paper_recap_min: int = int(paper_recap_min or getattr(config, "TELEGRAM_PAPER_RECAP_MIN", 60))
        self.live_alert_level: AlertLevel = (live_alert_level or getattr(config, "TELEGRAM_LIVE_ALERT_LEVEL", "normal"))  # type: ignore
        self.heartbeat_min: int = int(heartbeat_min or getattr(config, "TELEGRAM_HEARTBEAT_MIN", 10))

        # buffers / timers
        now = time.time()
        self._paper_buffer: List[TradeEvent] = []
        self._last_paper_flush_ts: float = now
        self._last_heartbeat_ts: float = 0.0

        # runtime stats to include in heartbeat
        self._last_price: Optional[float] = None
        self._last_symbol: Optional[str] = None
        self._last_equity: Optional[float] = None
        self._last_balance: Optional[float] = None

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def set_mode(self, mode: Mode):
        self.mode = mode

    def set_prefs(self, *, paper_recap_min: Optional[int] = None, live_alert_level: Optional[AlertLevel] = None, heartbeat_min: Optional[int] = None):
        if paper_recap_min is not None:
            self.paper_recap_min = max(5, int(paper_recap_min))
        if live_alert_level is not None:
            self.live_alert_level = live_alert_level
        if heartbeat_min is not None:
            self.heartbeat_min = max(1, int(heartbeat_min))

    def update_metrics_snapshot(self, *, price: Optional[float] = None, symbol: Optional[str] = None, equity: Optional[float] = None, balance: Optional[float] = None):
        """Let the bot drop some latest metrics so heartbeats/digests are informative."""
        if price is not None:
            self._last_price = float(price)
        if symbol is not None:
            self._last_symbol = symbol
        if equity is not None:
            self._last_equity = float(equity)
        if balance is not None:
            self._last_balance = float(balance)

    def notify_status(self, text: str):
        """One-off status/info line (thin wrapper)."""
        try:
            self.notifier.send_message({"level": "INFO", "message": text}, format="log")
        except Exception:
            logger.exception("notify_status failed")

    def notify_error(self, text: str):
        """Errors / warnings (always pass through)."""
        try:
            self.notifier.send_message({"type": "ERROR", "message": text}, format="alert")
        except Exception:
            logger.exception("notify_error failed")

    def notify_trade(self, evt: TradeEvent):
        """
        Trade-level notification.
          * Paper: buffer for recap.
          * Live: immediate according to alert level.
        """
        if self.mode == "paper":
            self._paper_buffer.append(evt)
            # Optional quick breadcrumb so you know it's active without spamming
            if self.live_alert_level == "verbose":
                self._emit_text(self._fmt_breadcrumb(evt))
            return

        # live
        level = self.live_alert_level
        if level == "quiet":
            # Only entries/exits (if pnl present assume close)
            if evt.pnl is not None or (evt.status or "").lower() in ("closed", "close"):
                self._emit_text(self._fmt_trade_line(evt))
            else:
                # entry opened — send minimal
                self._emit_text(self._fmt_breadcrumb(evt))
            return

        if level == "normal":
            # entries & exits with concise details
            self._emit_text(self._fmt_trade_line(evt))
            return

        # verbose
        self._emit_text(self._fmt_trade_line(evt, verbose=True))

    def tick(self, now: Optional[float] = None):
        """
        Call periodically (e.g., once each main loop iteration).
        Handles:
          - Paper recap flush
          - Heartbeat ping
        """
        t = now or time.time()

        # Heartbeat (both modes)
        if t - self._last_heartbeat_ts >= self.heartbeat_min * 60:
            self._send_heartbeat()
            self._last_heartbeat_ts = t

        # Paper recap
        if self.mode == "paper" and (t - self._last_paper_flush_ts) >= self.paper_recap_min * 60:
            self._flush_paper_digest()
            self._last_paper_flush_ts = t

    # ──────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────

    def _send_heartbeat(self):
        sym = self._last_symbol or getattr(config, "DEFAULT_SYMBOL", "BTC/USDT")
        px = self._last_price
        eq = self._last_equity
        bal = self._last_balance
        bits = ["🫀 <b>Heartbeat</b>"]
        bits.append(f"Mode: <code>{self.mode}</code> / Alerts: <code>{self.live_alert_level}</code>")
        bits.append(f"Symbol: <code>{sym}</code>")
        if px is not None:
            bits.append(f"Price: <code>{self._fmt_num(px)}</code>")
        if bal is not None:
            bits.append(f"Balance: <code>${self._fmt_num(bal)}</code>")
        if eq is not None:
            bits.append(f"Equity: <code>${self._fmt_num(eq)}</code>")
        self._emit_text("\n".join(bits))

    def _flush_paper_digest(self):
        if not self._paper_buffer:
            # still send a brief “no trades” message so you know it’s alive
            self._emit_text("📝 Paper recap: no trades in this interval.")
            return

        trades = self._paper_buffer
        self._paper_buffer = []  # reset

        # Aggregate
        n = len(trades)
        realized = [e for e in trades if e.pnl is not None]
        wins = [e for e in realized if (e.pnl or 0) > 0]
        losses = [e for e in realized if (e.pnl or 0) <= 0]

        total_pnl = sum((e.pnl or 0) for e in realized)
        avg_ret = _safe_avg([e.return_pct for e in realized if e.return_pct is not None])

        # Small symbol summary
        by_symbol: Dict[str, Dict[str, float]] = {}
        for e in realized:
            d = by_symbol.setdefault(e.symbol, {"count": 0, "pnl": 0.0})
            d["count"] += 1
            d["pnl"] += float(e.pnl or 0)

        lines = []
        lines.append("📝 <b>Paper Trading Recap</b>")
        lines.append(f"Trades: <b>{n}</b>  |  Realized: <b>{len(realized)}</b>  |  Win/Loss: <b>{len(wins)}/{len(losses)}</b>")
        lines.append(f"Net PnL: <b>${self._fmt_num(total_pnl)}</b>   Avg Return: <b>{self._fmt_num(avg_ret)}%</b>")

        if by_symbol:
            top = sorted(by_symbol.items(), key=lambda kv: kv[1]["pnl"], reverse=True)[:6]
            sym_chunk = " · ".join(f"{s}: ${self._fmt_num(v['pnl'])} ({int(v['count'])})" for s, v in top)
            lines.append(f"Top symbols: {sym_chunk}")

        # Show last 3 trades terse
        lines.append("\nRecent:")
        for e in trades[-3:]:
            lines.append("• " + self._fmt_trade_line(e, terse=True))

        self._emit_text("\n".join(lines))

    def _fmt_trade_line(self, e: TradeEvent, verbose: bool = False, terse: bool = False) -> str:
        # base
        side = e.side.upper()
        qty = self._fmt_num(e.qty, 6)
        px = self._fmt_num(e.price)
        sym = e.symbol
        pnl = e.pnl
        pct = e.return_pct
        lev = e.leverage
        status = (e.status or "").upper()

        if terse:
            tail = []
            if pnl is not None:
                tail.append(f"PnL ${self._fmt_num(pnl)}")
            if pct is not None:
                tail.append(f"{self._fmt_num(pct)}%")
            return f"{side} {sym} x{qty} @ {px}" + (" · " + " / ".join(tail) if tail else "")

        base = f"📊 <b>{side}</b> <code>{sym}</code> x<code>{qty}</code> @ <code>{px}</code>"
        if status:
            base += f"  [{status}]"
        tail = []
        if pnl is not None:
            tail.append(f"PnL <b>${self._fmt_num(pnl)}</b>")
        if pct is not None:
            tail.append(f"<b>{self._fmt_num(pct)}%</b>")
        if lev is not None:
            tail.append(f"{self._fmt_num(lev,2)}×")
        line = base + (("  →  " + " / ".join(tail)) if tail else "")

        if verbose:
            extra = []
            if e.opened:
                extra.append(f"Opened: {e.opened}")
            if e.closed:
                extra.append(f"Closed: {e.closed}")
            if e.meta:
                # include a couple of interesting meta fields if present
                for k in ("tp", "sl", "reason", "signal"):
                    if k in e.meta:
                        extra.append(f"{k.upper()}: {e.meta[k]}")
            if extra:
                line += "\n" + " ; ".join(extra)
        return line

    def _fmt_breadcrumb(self, e: TradeEvent) -> str:
        return f"🔎 {e.side.upper()} {e.symbol} x{self._fmt_num(e.qty, 6)} @ {self._fmt_num(e.price)}"

    def _emit_text(self, text: str):
        try:
            self.notifier.send_message(text, format="text")
        except Exception:
            logger.exception("emit_text failed")

    @staticmethod
    def _fmt_num(x: float, prec: int = 4) -> str:
        try:
            if x is None or math.isnan(float(x)) or math.isinf(float(x)):
                return "0"
            # prettier formatting for big/small values
            if abs(x) >= 1000:
                return f"{x:,.2f}"
            return f"{x:.{prec}f}"
        except Exception:
            return str(x)


def _safe_avg(vals: List[Optional[float]]) -> float:
    clean = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    return (sum(clean) / len(clean)) if clean else 0.0

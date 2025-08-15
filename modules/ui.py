# modules/ui.py

import asyncio
from datetime import datetime
from typing import Callable, Dict, Any, Optional

from textual.app import App, ComposeResult
from textual.widgets import Button, Header, Footer, Log, Static, Sparkline
from textual.containers import Horizontal, Vertical


class TradingUI(App):
    """A Textual-based UI for the Trading Bot."""

    CSS_PATH = "tui.css"
    TITLE = "Textual Trading Bot"

    # Keep a reference to the bot instance
    bot: Any = None
    _action_handlers: Dict[str, Callable[[], None]] = {}

    def __init__(self, bot: Any, **kwargs):
        """Initializes the TradingUI."""
        super().__init__(**kwargs)
        self.bot = bot
        self._wallet_hist_data = [0.0] * 50
        self._vwallet_hist_data = [0.0] * 50
        self._points_hist_data = [0.0] * 50

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        with Horizontal(id="main-container"):
            with Vertical(id="left-panel"):
                yield Static("Controls", classes="panel-title")
                yield Button(
                    "▶ Start Training", id="start_training", variant="success")
                yield Button(
                    "⏹ Stop Training", id="stop_training", variant="error")
                yield Button(
                    "▶ Start Trading", id="start_trading", variant="success")
                yield Button(
                    "⏹ Stop Trading", id="stop_trading", variant="error")
            with Vertical(id="center-panel"):
                with Horizontal(id="status-bar"):
                    yield Static("Disconnected", id="connection-status")
                    yield Static("Idle", id="bot-mode")
                    yield Static("--", id="heartbeat-status")
                    yield Static("", id="clock")
                yield Log(id="log-view", auto_scroll=True)
            with Vertical(id="right-panel"):
                yield Static("Live Metrics", classes="panel-title")
                yield Static("$0.00", id="wallet-balance")
                yield Static("$0.00", id="portfolio-value")
                yield Static("N/A", id="current-symbol")
                yield Static("N/A", id="current-timeframe")
                yield Static("\nWallet Balance History", classes="panel-title")
                yield Sparkline(self._wallet_hist_data, id="wallet-sparkline")
                yield Static("Portfolio Value History", classes="panel-title")
                yield Sparkline(
                    self._vwallet_hist_data, id="vwallet-sparkline")
                yield Static("Reward Points History", classes="panel-title")
                yield Sparkline(
                    self._points_hist_data, id="points-sparkline")
        yield Footer()

    def on_mount(self) -> None:
        """Called when the app is mounted."""
        self.set_interval(1, self._update_clock)
        self.set_interval(1, self._refresh_bot_status)

    def _update_clock(self) -> None:
        """Updates the clock in the status bar."""
        self.query_one("#clock", Static).update(
            datetime.now().strftime("%H:%M:%S"))

    def add_action_handler(self, name: str, callback: Callable[[], None]):
        """Adds an action handler for a button."""
        self._action_handlers[name] = callback

    def set_title(self, title: str):
        """Sets the title of the UI."""
        self.title = title

    async def log(self, message: str, level: str = 'INFO'):
        """Logs a message to the UI."""
        if self.is_running:
            log_widget = self.query_one("#log-view", Log)
            await self.call_soon(log_widget.write, f"[{level}] {message}")

    def _update_live_metrics(self, metrics: Dict[str, Any]):
        if (bal := metrics.get("balance")) is not None:
            self.query_one("#wallet-balance", Static).update(f"${bal:,.2f}")
        if (eq := metrics.get("equity")) is not None:
            self.query_one("#portfolio-value", Static).update(f"${eq:,.2f}")
        if (sym := metrics.get("symbol")) is not None:
            self.query_one("#current-symbol", Static).update(sym)
        if (tf := metrics.get("timeframe")) is not None:
            self.query_one("#current-timeframe", Static).update(tf)

    def update_live_metrics(self, metrics: Dict[str, Any]):
        """Updates the live metrics in the UI."""
        if self.is_running:
            self.call_soon(self._update_live_metrics, metrics)

    def _update_timeseries(self, *, wallet: Optional[float] = None,
                           vwallet: Optional[float] = None,
                           points: Optional[float] = None):
        if wallet is not None:
            self._wallet_hist_data = self._wallet_hist_data[1:] + [wallet]
            self.query_one(
                "#wallet-sparkline", Sparkline).data = self._wallet_hist_data
        if vwallet is not None:
            self._vwallet_hist_data = self._vwallet_hist_data[1:] + [vwallet]
            self.query_one(
                "#vwallet-sparkline", Sparkline).data = self._vwallet_hist_data
        if points is not None:
            self._points_hist_data = self._points_hist_data[1:] + [points]
            self.query_one(
                "#points-sparkline", Sparkline).data = self._points_hist_data

    def update_timeseries(self, *, wallet: Optional[float] = None,
                          vwallet: Optional[float] = None,
                          points: Optional[float] = None):
        """Updates the timeseries data in the UI."""
        if self.is_running:
            self.call_soon(
                self._update_timeseries,
                wallet=wallet, vwallet=vwallet, points=points)

    def set_button_active(self, name: str):
        """Sets a button to active state."""
        async def _set_button_active():
            await self.log(f"Action '{name}' activated.", level="SUCCESS")
        if self.is_running:
            asyncio.run_coroutine_threadsafe(_set_button_active(), self._loop)

    def _refresh_bot_status(self) -> None:
        if not self.bot:
            return
        is_conn = bool(getattr(self.bot, "is_connected", True))
        self.query_one("#connection-status", Static).update(
            f"⚡ {'Connected' if is_conn else 'Disconnected'}")
        is_training = bool(getattr(self.bot, "training", False))
        self.query_one("#bot-mode", Static).update(
            f"Mode: {'Training' if is_training else 'Live' if getattr(self.bot, 'trading', False) else 'Idle'}")
        if hb_ts := getattr(self.bot, "last_heartbeat", None):
            self.query_one("#heartbeat-status", Static).update(
                f"Heartbeat: {datetime.fromtimestamp(hb_ts).strftime('%H:%M:%S')}")
        else:
            self.query_one("#heartbeat-status", Static).update("Heartbeat: --")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Called when a button is pressed."""
        action_id = event.button.id
        if action_id in self._action_handlers:
            try:
                self._action_handlers[action_id]()
            except Exception as e:
                asyncio.run(
                    self.log(f"Action '{action_id}' failed: {e}", level='ERROR'))
        else:
            asyncio.run(
                self.log(f"No handler for action '{action_id}'", level='WARN'))

    def run_ui(self):
        """Runs the UI."""
        self.run()

    @property
    def is_running(self) -> bool:
        """Returns whether the UI is running."""
        return self._loop is not None

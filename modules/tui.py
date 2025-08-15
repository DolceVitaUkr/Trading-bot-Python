# modules/tui.py

import asyncio
from datetime import datetime
from typing import Callable, Dict, Any, Optional, List, Tuple

from rich.panel import Panel
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Header, Footer, Log, Static, Sparkline
from textual.reactive import reactive

class TradingUI(App):
    """A Textual-based UI for the Trading Bot."""

    CSS_PATH = "tui.css"

    # Keep a reference to the bot instance
    bot: Any = None
    _action_handlers: Dict[str, Callable[[], None]] = {}

    # --- Reactive properties for updating the UI ---
    connection_status = reactive("⚡ Disconnected")
    bot_mode = reactive("Mode: Idle")
    last_heartbeat = reactive("Heartbeat: --")
    clock = reactive("")

    wallet_balance = reactive("$0.00")
    portfolio_value = reactive("$0.00")
    current_symbol = reactive("N/A")
    current_timeframe = reactive("N/A")

    def __init__(self, bot: Any, **kwargs):
        super().__init__(**kwargs)
        self.bot = bot
        self._wallet_hist_data = [0.0] * 50
        self._vwallet_hist_data = [0.0] * 50
        self._points_hist_data = [0.0] * 50

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        # Header
        yield Header()

        # Main Content
        with Horizontal(id="main-container"):
            # Left panel for controls
            with Vertical(id="left-panel"):
                yield Static("Controls", classes="panel-title")
                yield Button("▶ Start Training", id="start_training", variant="success")
                yield Button("⏹ Stop Training", id="stop_training", variant="error")
                yield Button("▶ Start Trading", id="start_trading", variant="success")
                yield Button("⏹ Stop Trading", id="stop_trading", variant="error")
                # TODO: Add notification settings if needed

            # Center panel for logs and metrics
            with Vertical(id="center-panel"):
                with Horizontal(id="status-bar"):
                    yield Static(self.connection_status, id="connection-status")
                    yield Static(self.bot_mode, id="bot-mode")
                    yield Static(self.last_heartbeat, id="heartbeat-status")
                    yield Static(self.clock, id="clock")

                yield Log(id="log-view", auto_scroll=True)

            # Right panel for metrics and charts
            with Vertical(id="right-panel"):
                yield Static("Live Metrics", classes="panel-title")
                yield Static(self.wallet_balance, id="wallet-balance")
                yield Static(self.portfolio_value, id="portfolio-value")
                yield Static(f"Symbol: {self.current_symbol}", id="current-symbol")
                yield Static(f"Timeframe: {self.current_timeframe}", id="current-timeframe")

                yield Static("\nWallet Balance History", classes="panel-title")
                yield Sparkline(self._wallet_hist_data, id="wallet-sparkline")
                yield Static("Portfolio Value History", classes="panel-title")
                yield Sparkline(self._vwallet_hist_data, id="vwallet-sparkline")
                yield Static("Reward Points History", classes="panel-title")
                yield Sparkline(self._points_hist_data, id="points-sparkline")

        # Footer
        yield Footer()

    def on_mount(self) -> None:
        """Called when the app is mounted."""
        self.set_interval(1, self._update_clock)
        self.set_interval(1, self._refresh_bot_status)

    def _update_clock(self) -> None:
        """Updates the clock in the status bar."""
        self.query_one("#clock", Static).update(datetime.now().strftime("%H:%M:%S"))

    # --- Watch methods for reactive properties ---

    def watch_connection_status(self, status: str) -> None:
        self.query_one("#connection-status", Static).update(status)

    def watch_bot_mode(self, mode: str) -> None:
        self.query_one("#bot-mode", Static).update(mode)

    def watch_last_heartbeat(self, hb: str) -> None:
        self.query_one("#heartbeat-status", Static).update(hb)

    def watch_wallet_balance(self, balance: str) -> None:
        self.query_one("#wallet-balance", Static).update(f"Wallet: {balance}")

    def watch_portfolio_value(self, value: str) -> None:
        self.query_one("#portfolio-value", Static).update(f"Portfolio: {value}")

    def watch_current_symbol(self, symbol: str) -> None:
        self.query_one("#current-symbol", Static).update(f"Symbol: {symbol}")

    def watch_current_timeframe(self, tf: str) -> None:
        self.query_one("#current-timeframe", Static).update(f"Timeframe: {tf}")


    # --- Public API for the bot ---

    def add_action_handler(self, name: str, callback: Callable[[], None]):
        self._action_handlers[name] = callback

    def set_title(self, title: str):
        # Textual sets the title in the App constructor, but we can update it
        self.title = title

    def log(self, message: str, level: str = 'INFO'):
        async def _log():
            log_widget = self.query_one("#log-view", Log)
            log_widget.write(f"[{level}] {message}")
        asyncio.run_coroutine_threadsafe(_log(), self.loop)

    def update_live_metrics(self, metrics: Dict[str, Any]):
        def _update():
            if (bal := metrics.get("balance")) is not None:
                self.wallet_balance = f"${bal:,.2f}"
            if (eq := metrics.get("equity")) is not None:
                self.portfolio_value = f"${eq:,.2f}"
            if (sym := metrics.get("symbol")) is not None:
                self.current_symbol = sym
            if (tf := metrics.get("timeframe")) is not None:
                self.current_timeframe = tf
        self.call_from_thread(_update)

    def update_timeseries(self, *, wallet: Optional[float] = None,
                          vwallet: Optional[float] = None, points: Optional[float] = None):
        def _update():
            if wallet is not None:
                self._wallet_hist_data = self._wallet_hist_data[1:] + [wallet]
                self.query_one("#wallet-sparkline", Sparkline).data = self._wallet_hist_data
            if vwallet is not None:
                self._vwallet_hist_data = self._vwallet_hist_data[1:] + [vwallet]
                self.query_one("#vwallet-sparkline", Sparkline).data = self._vwallet_hist_data
            if points is not None:
                self._points_hist_data = self._points_hist_data[1:] + [points]
                self.query_one("#points-sparkline", Sparkline).data = self._points_hist_data
        self.call_from_thread(_update)

    def set_button_active(self, name: str):
        # In Textual, we might indicate activity differently, e.g., log it
        self.log(f"Action '{name}' activated.", level="SUCCESS")

    def _refresh_bot_status(self) -> None:
        """Periodically refresh status from the bot object."""
        if not self.bot:
            return

        is_conn = bool(getattr(self.bot, "is_connected", True))
        self.connection_status = f"⚡ {'Connected' if is_conn else 'Disconnected'}"

        is_training = bool(getattr(self.bot, "training", False))
        self.bot_mode = f"Mode: {'Training' if is_training else 'Live' if getattr(self.bot, 'trading', False) else 'Idle'}"

        if hb_ts := getattr(self.bot, "last_heartbeat", None):
            self.last_heartbeat = f"Heartbeat: {datetime.fromtimestamp(hb_ts).strftime('%H:%M:%S')}"
        else:
            self.last_heartbeat = "Heartbeat: --"

    # --- Event Handlers ---

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        action_id = event.button.id
        if action_id in self._action_handlers:
            try:
                self._action_handlers[action_id]()
            except Exception as e:
                self.log(f"Action '{action_id}' failed: {e}", level='ERROR')
        else:
            self.log(f"No handler for action '{action_id}'", level='WARN')

    def run_ui(self):
        """Run the Textual app. This replaces the old run() method."""
        self.run()

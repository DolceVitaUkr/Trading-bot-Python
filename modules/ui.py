# modules/ui.py

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import time
from typing import Callable, Dict, Any, Optional, List
from collections import deque
from datetime import datetime

import config

UI_REFRESH_INTERVAL = getattr(config, "UI_REFRESH_INTERVAL", 1000)


class TradingUI:
    """
    Reworked UI:
      - Three charts (Wallet, Virtual Wallet, Reward Points)
      - Buttons change color/disabled state on press
      - Same public API as before (add_action_handler, update_* methods, log, run)
    """

    def __init__(self, bot: Any):
        self.bot = bot
        self._action_handlers: Dict[str, Callable[[], None]] = {}

        # Simple in-memory time series (last 500 points)
        self.max_points = 500
        self.ts_wallet = deque(maxlen=self.max_points)   # (t, val)
        self.ts_virtual = deque(maxlen=self.max_points)  # (t, val)
        self.ts_points = deque(maxlen=self.max_points)   # (t, val)

        self.root = tk.Tk()
        self.root.title("AI Trading Terminal")
        self.root.geometry("1600x900")
        self._configure_style()

        self._create_status_bar()
        self._create_left_controls()
        self._create_center_charts()
        self._create_right_metrics()
        self._create_bottom_logs()

        self.root.after(UI_REFRESH_INTERVAL, self._refresh)

    # ─────────────────────────────────────────────────────────────────────
    # Layout & Style
    # ─────────────────────────────────────────────────────────────────────

    def _configure_style(self):
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except Exception:
            pass
        style.configure('.', background='#1e1e1e', foreground='white')
        style.configure('TLabel', background='#1e1e1e', foreground='white')
        style.configure('TFrame', background='#1e1e1e')
        style.configure('TLabelframe', background='#1e1e1e', foreground='white')
        style.configure('TLabelframe.Label', background='#1e1e1e', foreground='white')
        style.configure('TButton', background='#2a2a2a', foreground='white')
        style.map('TButton',
                  background=[('active', '#3a3a3a')],
                  relief=[('pressed', 'sunken')])

        # Button styles for on/off states
        style.configure('Active.TButton', background='#1f6feb', foreground='white')   # blue
        style.configure('Stop.TButton',   background='#d73a49', foreground='white')   # red
        style.configure('Idle.TButton',   background='#2a2a2a', foreground='white')   # default

    def _create_status_bar(self):
        frame = ttk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)

        self.conn_label = ttk.Label(frame, text="⚡ Disconnected")
        self.conn_label.pack(side=tk.LEFT, padx=10)

        self.mode_label = ttk.Label(frame, text="Mode: Idle")
        self.mode_label.pack(side=tk.LEFT, padx=10)

        self.heartbeat_label = ttk.Label(frame, text="Heartbeat: --")
        self.heartbeat_label.pack(side=tk.LEFT, padx=10)

        self.time_label = ttk.Label(frame, text=time.strftime('%H:%M:%S'))
        self.time_label.pack(side=tk.RIGHT, padx=10)

    def _create_left_controls(self):
        frame = ttk.Labelframe(self.root, text="Controls", padding=10)
        frame.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)

        # Buttons with persistent references to change styles later
        self.btn_start_training = ttk.Button(frame, text="▶ Start Training",
                                             command=lambda: self._on_button("start_training", "training"))
        self.btn_stop_training = ttk.Button(frame, text="⏹ Stop Training",
                                            command=lambda: self._on_button("stop_training", "idle"))

        self.btn_start_trading = ttk.Button(frame, text="▶ Start Trading",
                                            command=lambda: self._on_button("start_trading", "trading"))
        self.btn_stop_trading = ttk.Button(frame, text="⏹ Stop Trading",
                                           command=lambda: self._on_button("stop_trading", "idle"))

        self.btn_start_training.pack(fill=tk.X, pady=4)
        self.btn_stop_training.pack(fill=tk.X, pady=4)

        ttk.Separator(frame, orient='horizontal').pack(fill=tk.X, pady=6)

        self.btn_start_trading.pack(fill=tk.X, pady=8)
        self.btn_stop_trading.pack(fill=tk.X, pady=4)

        ttk.Separator(frame, orient='horizontal').pack(fill=tk.X, pady=10)

        notif = ttk.Labelframe(frame, text="Notification Settings", padding=8)
        notif.pack(fill=tk.X, pady=6)

        ttk.Label(notif, text="Paper recap (min):").pack(anchor=tk.W)
        self.paper_interval_var = tk.IntVar(value=getattr(config, "TELEGRAM_PAPER_RECAP_MIN", 60))
        tk.Spinbox(notif, from_=5, to=240, increment=5, textvariable=self.paper_interval_var, width=6).pack(anchor=tk.W, pady=2)

        ttk.Label(notif, text="Live alerts:").pack(anchor=tk.W, pady=(6, 0))
        self.live_alert_var = tk.StringVar(value=getattr(config, "TELEGRAM_LIVE_ALERT_LEVEL", "normal"))
        ttk.Combobox(
            notif,
            textvariable=self.live_alert_var,
            values=["quiet", "normal", "verbose"],
            state="readonly",
            width=10
        ).pack(anchor=tk.W, pady=2)

        ttk.Button(notif, text="Apply", command=self._apply_notification_prefs).pack(fill=tk.X, pady=(8, 0))

    def _create_center_charts(self):
        # Try to set up matplotlib
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except Exception:
            self.wallet_ax = self.virtual_ax = self.points_ax = None
            self.chart_canvas = None
            frame = ttk.Labelframe(self.root, text="Charts (matplotlib not available)", padding=10)
            frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)
            return

        frame = ttk.Labelframe(self.root, text="Performance Charts")
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        fig = Figure(figsize=(10, 6), dpi=100)

        # Three separate axes stacked vertically
        self.wallet_ax = fig.add_subplot(311)
        self.virtual_ax = fig.add_subplot(312)
        self.points_ax = fig.add_subplot(313)

        for ax, title in [
            (self.wallet_ax, "Wallet Balance ($)"),
            (self.virtual_ax, "Virtual Wallet ($)"),
            (self.points_ax, "Reward Points"),
        ]:
            ax.grid(True, alpha=0.2)
            ax.set_title(title)

        self.chart_canvas = FigureCanvasTkAgg(fig, master=frame)
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _create_right_metrics(self):
        frame = ttk.Labelframe(self.root, text="Metrics", padding=10)
        frame.pack(side=tk.RIGHT, fill=tk.Y, padx=6, pady=6)

        ttk.Label(frame, text="Wallet Balance:").pack(anchor=tk.W)
        self.balance_var = tk.StringVar(value="$0.00")
        ttk.Label(frame, textvariable=self.balance_var).pack(anchor=tk.W, pady=2)

        ttk.Label(frame, text="Portfolio Value:").pack(anchor=tk.W, pady=(10, 0))
        self.value_var = tk.StringVar(value="$0.00")
        ttk.Label(frame, textvariable=self.value_var).pack(anchor=tk.W, pady=2)

        ttk.Label(frame, text="Current Symbol:").pack(anchor=tk.W, pady=(10, 0))
        self.symbol_var = tk.StringVar(value="N/A")
        ttk.Label(frame, textvariable=self.symbol_var).pack(anchor=tk.W, pady=2)

        ttk.Label(frame, text="Timeframe:").pack(anchor=tk.W, pady=(10, 0))
        self.tf_var = tk.StringVar(value="N/A")
        ttk.Label(frame, textvariable=self.tf_var).pack(anchor=tk.W, pady=2)

    def _create_bottom_logs(self):
        frame = ttk.Labelframe(self.root, text="Logs")
        frame.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=6)
        self.log_text = scrolledtext.ScrolledText(frame, height=10, bg='#2d2d2d', fg='white')
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.tag_config('INFO', foreground='#00ff99')
        self.log_text.tag_config('ERROR', foreground='#ff3300')
        self.log_text.tag_config('SUCCESS', foreground='#00ccff')
        self.log_text.tag_config('WARN', foreground='#ffd166')

    # ─────────────────────────────────────────────────────────────────────
    # Public UI API (same names as before)
    # ─────────────────────────────────────────────────────────────────────

    def add_action_handler(self, name: str, callback: Callable[[], None]):
        self._action_handlers[name] = callback

    def set_title(self, title: str):
        self.root.title(title)

    def update_simulation_results(self, balance: float = None, points: float = None, **kwargs):
        """
        Accepts either (balance, points) or legacy (final_balance, total_points).
        Also appends to charts.
        """
        if balance is None:
            balance = kwargs.get("final_balance", 0.0)
        if points is None:
            points = kwargs.get("total_points", 0.0)

        self.balance_var.set(f"${balance:,.2f}")
        self._append_virtual(balance)
        self._append_points(points)
        self._redraw_charts()
        self.log(f"Simulation complete: Balance=${balance:,.2f}, Points={points:.2f}", level='SUCCESS')

    def update_live_metrics(self, metrics: Dict[str, Any]):
        bal = metrics.get("balance")
        if bal is not None:
            self.balance_var.set(f"${bal:,.2f}")
            self._append_wallet(float(bal))

        eq = metrics.get("equity")
        if eq is not None:
            self.value_var.set(f"${eq:,.2f}")

        sym = metrics.get("symbol")
        if sym:
            self.symbol_var.set(sym)

        tf = metrics.get("timeframe")
        if tf:
            self.tf_var.set(tf)

        self._redraw_charts()

    def log(self, message: str, level: str = 'INFO'):
        ts = time.strftime('%H:%M:%S')
        tag = level.upper()
        if tag not in ('INFO', 'ERROR', 'SUCCESS', 'WARN'):
            tag = 'INFO'
        self.log_text.insert(tk.END, f"{ts} - {message}\n", tag)
        self.log_text.see(tk.END)

    async def shutdown(self):
        pass

    def run(self):
        self.root.mainloop()

    # ─────────────────────────────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────────────────────────────

    def _invoke(self, action: str):
        cb = self._action_handlers.get(action)
        if not cb:
            messagebox.showwarning("Not implemented", f"No handler for '{action}'")
            return
        try:
            cb()
        except Exception as e:
            self.log(f"Action '{action}' failed: {e}", level='ERROR')

    def _apply_notification_prefs(self):
        prefs = {
            "paper_recap_minutes": int(self.paper_interval_var.get()),
            "live_alert_level": self.live_alert_var.get(),
        }
        if hasattr(self.bot, "apply_notification_prefs") and callable(getattr(self.bot, "apply_notification_prefs")):
            try:
                self.bot.apply_notification_prefs(prefs)
                self.log(f"Notification prefs applied: {prefs}", level='SUCCESS')
            except Exception as e:
                self.log(f"Failed to apply notification prefs: {e}", level='ERROR')
        else:
            self.log("Bot has no 'apply_notification_prefs' hook (skipped).", level='WARN')

    def _refresh(self):
        # Clock
        self.time_label.config(text=time.strftime('%H:%M:%S'))

        # Connection & Mode hints from bot (if available)
        is_conn = bool(getattr(self.bot, "is_connected", False))
        self.conn_label.config(text=f"⚡ {'Connected' if is_conn else 'Disconnected'}")

        is_training = bool(getattr(self.bot, "is_training", False))
        is_trading = bool(getattr(self.bot, "is_trading", False))
        mode = "Training" if is_training else "Live" if is_trading else "Idle"
        self.mode_label.config(text=f"Mode: {mode}")

        hb_ts = getattr(self.bot, "last_heartbeat", None)
        if hb_ts:
            self.heartbeat_label.config(text=f"Heartbeat: {time.strftime('%H:%M:%S', time.localtime(hb_ts))}")
        else:
            self.heartbeat_label.config(text="Heartbeat: --")

        # Lightweight chart keeps redrawing occasionally
        self._redraw_charts()

        self.root.after(UI_REFRESH_INTERVAL, self._refresh)

    def _on_button(self, action: str, state: str):
        """Change button styles to reflect current state and invoke action."""
        # Visual state
        if action == "start_training":
            self._set_button_states(active=self.btn_start_training)
        elif action == "stop_training":
            self._set_button_states(active=None)
        elif action == "start_trading":
            self._set_button_states(active=self.btn_start_trading)
        elif action == "stop_trading":
            self._set_button_states(active=None)

        self._invoke(action)

    def _set_button_states(self, active: Optional[ttk.Button]):
        # Reset styles
        for b in [self.btn_start_training, self.btn_stop_training, self.btn_start_trading, self.btn_stop_trading]:
            b.configure(style='Idle.TButton')
            b.state(["!disabled"])

        if active is self.btn_start_training:
            self.btn_start_training.configure(style='Active.TButton')
            self.btn_start_trading.configure(style='Idle.TButton')
        elif active is self.btn_start_trading:
            self.btn_start_trading.configure(style='Active.TButton')
            self.btn_start_training.configure(style='Idle.TButton')
        else:
            # Stop pressed -> highlight stop
            self.btn_stop_training.configure(style='Stop.TButton')
            self.btn_stop_trading.configure(style='Stop.TButton')

    # --- Chart helpers ---

    def _append_wallet(self, value: float):
        self.ts_wallet.append((time.time(), float(value)))

    def _append_virtual(self, value: float):
        self.ts_virtual.append((time.time(), float(value)))

    def _append_points(self, value: float):
        self.ts_points.append((time.time(), float(value)))

    def _redraw_charts(self):
        if not all([self.wallet_ax, self.virtual_ax, self.points_ax, self.chart_canvas]):
            return

        def plot_series(ax, series: deque, ylabel: str):
            ax.clear()
            ax.grid(True, alpha=0.2)
            ax.set_title(ylabel)
            if series:
                xs = [datetime.fromtimestamp(t) for t, _ in series]
                ys = [v for _, v in series]
                ax.plot(xs, ys)
            # Keep labels light-weight to avoid CPU spike

        plot_series(self.wallet_ax, self.ts_wallet, "Wallet Balance ($)")
        plot_series(self.virtual_ax, self.ts_virtual, "Virtual Wallet ($)")
        plot_series(self.points_ax, self.ts_points, "Reward Points")

        self.chart_canvas.draw_idle()

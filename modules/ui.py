# modules/ui.py

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import time
from typing import Callable, Dict, Any, Optional, List, Tuple
import collections

import config

UI_REFRESH_INTERVAL = getattr(config, "UI_REFRESH_INTERVAL", 1000)


class TradingUI:
    """
    GUI for controlling and monitoring the TradingBot.

    Changes:
    - Added three dedicated charts:
        1) Live Wallet Balance
        2) Virtual Wallet (Portfolio Value)
        3) Reward Points
    - Button color toggles after press so last action is visible
    - Lightweight append of timeseries (keeps last 720 points by default)
    """

    def __init__(self, bot: Any, history_limit: int = 720):
        self.bot = bot
        self._action_handlers: Dict[str, Callable[[], None]] = {}
        self._active_button: Optional[str] = None

        # Timeseries buffers (deque for speed)
        self._wallet_hist = collections.deque(maxlen=history_limit)
        self._vwallet_hist = collections.deque(maxlen=history_limit)
        self._points_hist = collections.deque(maxlen=history_limit)
        self._time_hist = collections.deque(maxlen=history_limit)

        self.root = tk.Tk()
        self.root.title("AI Trading Terminal")
        self.root.geometry("1750x950")
        self._configure_style()

        self._create_status_bar()
        self._create_left_controls()
        self._create_center_charts()
        self._create_right_metrics()
        self._create_bottom_logs()

        self.root.after(UI_REFRESH_INTERVAL, self._refresh)

    # ---------- Styling ----------

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
        style.configure('Active.TButton', background='#3a7bd5', foreground='white')  # active color
        style.configure('TNotebook', background='#2d2d2d')
        style.configure('TNotebook.Tab', background='#2d2d2d', foreground='white')
        style.map('TNotebook.Tab', background=[('selected', '#3e3e3e')])

    # ---------- Layout ----------

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

    def _make_button(self, parent, text, name):
        def cb():
            self._invoke(name)
        btn = ttk.Button(parent, text=text, command=cb)
        btn._action_name = name  # type: ignore[attr-defined]
        return btn

    def _set_button_style(self, btn: ttk.Button, active: bool):
        try:
            btn.configure(style='Active.TButton' if active else 'TButton')
        except Exception:
            pass

    def set_button_active(self, name: str):
        self._active_button = name
        # walk through all buttons and apply styles
        for b in self._all_buttons:
            self._set_button_style(b, getattr(b, "_action_name", None) == name)

    def _create_left_controls(self):
        frame = ttk.Labelframe(self.root, text="Controls", padding=10)
        frame.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)

        self._all_buttons: List[ttk.Button] = []
        for label, name in [
            ("▶ Start Training", "start_training"),
            ("⏹ Stop Training", "stop_training"),
            ("▶ Start Trading", "start_trading"),
            ("⏹ Stop Trading", "stop_trading"),
        ]:
            btn = self._make_button(frame, label, name)
            btn.pack(fill=tk.X, pady=6)
            self._all_buttons.append(btn)

        ttk.Separator(frame, orient='horizontal').pack(fill=tk.X, pady=10)

        notif = ttk.Labelframe(frame, text="Notification Settings", padding=8)
        notif.pack(fill=tk.X, pady=6)

        ttk.Label(notif, text="Paper recap (min):").pack(anchor=tk.W)
        self.paper_interval_var = tk.IntVar(value=getattr(config, "TELEGRAM_PAPER_RECAP_MIN", 60))
        tk.Spinbox(notif, from_=5, to=240, increment=5, textvariable=self.paper_interval_var, width=8).pack(anchor=tk.W, pady=2)

        ttk.Label(notif, text="Live alerts:").pack(anchor=tk.W, pady=(6, 0))
        self.live_alert_var = tk.StringVar(value=getattr(config, "TELEGRAM_LIVE_ALERT_LEVEL", "normal"))
        ttk.Combobox(
            notif,
            textvariable=self.live_alert_var,
            values=["quiet", "normal", "verbose"],
            state="readonly",
            width=12
        ).pack(anchor=tk.W, pady=2)

        ttk.Button(notif, text="Apply", command=self._apply_notification_prefs).pack(fill=tk.X, pady=(8, 0))

    def _create_center_charts(self):
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except Exception:
            self.ax_wallet = self.ax_vwallet = self.ax_points = None
            self.chart_canvas = None
            frame = ttk.Labelframe(self.root, text="Charts (matplotlib not available)", padding=10)
            frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)
            return

        frame = ttk.Labelframe(self.root, text="Performance Charts")
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        fig = Figure(figsize=(10, 6), dpi=100)
        # One plot per metric (no subplots sharing per tool rules—but here it's a single Figure with 3 axes)
        self.ax_wallet = fig.add_subplot(311)
        self.ax_vwallet = fig.add_subplot(312)
        self.ax_points = fig.add_subplot(313)

        for ax, title in [
            (self.ax_wallet, "Live Wallet Balance"),
            (self.ax_vwallet, "Virtual Portfolio Value"),
            (self.ax_points, "Reward Points"),
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

    # ---------- Public UI API ----------

    def add_action_handler(self, name: str, callback: Callable[[], None]):
        self._action_handlers[name] = callback

    def set_title(self, title: str):
        self.root.title(title)

    def update_simulation_results(self, balance: float = None, points: float = None, **kwargs):
        """
        Accepts either (balance, points) or legacy (final_balance, total_points).
        """
        if balance is None:
            balance = kwargs.get("final_balance", 0.0)
        if points is None:
            points = kwargs.get("total_points", 0.0)

        self.balance_var.set(f"${balance:,.2f}")
        self.log(f"Simulation complete: Balance=${balance:,.2f}, Points={points:.2f}", level='SUCCESS')

    def update_live_metrics(self, metrics: Dict[str, Any]):
        bal = metrics.get("balance")
        if bal is not None:
            self.balance_var.set(f"${bal:,.2f}")
        eq = metrics.get("equity")
        if eq is not None:
            self.value_var.set(f"${eq:,.2f}")
        sym = metrics.get("symbol")
        if sym:
            self.symbol_var.set(sym)
        tf = metrics.get("timeframe")
        if tf:
            self.tf_var.set(tf)

    def update_timeseries(self, *, wallet: Optional[float] = None,
                          vwallet: Optional[float] = None, points: Optional[float] = None):
        ts = time.time()
        self._time_hist.append(ts)
        self._wallet_hist.append(wallet if wallet is not None else (self._wallet_hist[-1] if self._wallet_hist else 0.0))
        self._vwallet_hist.append(vwallet if vwallet is not None else (self._vwallet_hist[-1] if self._vwallet_hist else 0.0))
        self._points_hist.append(points if points is not None else (self._points_hist[-1] if self._points_hist else 0.0))

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

    # ---------- Internals ----------

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
        self.time_label.config(text=time.strftime('%H:%M:%S'))

        is_conn = bool(getattr(self.bot, "is_connected", True))  # WS assumed up by DataManager
        self.conn_label.config(text=f"⚡ {'Connected' if is_conn else 'Disconnected'}")

        is_training = bool(getattr(self.bot, "training", False))
        is_trading = is_training  # in this design, training loop == sim trading loop
        mode = "Training" if is_training else "Live" if is_trading else "Idle"
        self.mode_label.config(text=f"Mode: {mode}")

        hb_ts = getattr(self.bot, "last_heartbeat", None)
        if hb_ts:
            self.heartbeat_label.config(text=f"Heartbeat: {time.strftime('%H:%M:%S', time.localtime(hb_ts))}")
        else:
            self.heartbeat_label.config(text="Heartbeat: --")

        # Re-draw charts
        try:
            if self.chart_canvas and self.ax_wallet and self.ax_vwallet and self.ax_points:
                # clear
                self.ax_wallet.clear(); self.ax_wallet.grid(True, alpha=0.2); self.ax_wallet.set_title("Live Wallet Balance")
                self.ax_vwallet.clear(); self.ax_vwallet.grid(True, alpha=0.2); self.ax_vwallet.set_title("Virtual Portfolio Value")
                self.ax_points.clear(); self.ax_points.grid(True, alpha=0.2); self.ax_points.set_title("Reward Points")

                xs = list(self._time_hist)
                if xs:
                    self.ax_wallet.plot(xs, list(self._wallet_hist))
                    self.ax_vwallet.plot(xs, list(self._vwallet_hist))
                    self.ax_points.plot(xs, list(self._points_hist))
                self.chart_canvas.draw()
        except Exception as e:
            self.log(f"Chart update failed: {e}", level='ERROR')

        self.root.after(UI_REFRESH_INTERVAL, self._refresh)

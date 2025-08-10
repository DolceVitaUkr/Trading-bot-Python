# modules/ui.py

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import time
from typing import Callable, Dict, Any, Optional, List

import config

UI_REFRESH_INTERVAL = getattr(config, "UI_REFRESH_INTERVAL", 1000)


class TradingUI:
    """
    GUI for controlling and monitoring the TradingBot.

    Updates:
      - Three charts (wallet balance, virtual wallet balance, reward points)
      - Button color feedback (last-pressed button highlighted)
      - Status bar shows last action
    """

    def __init__(self, bot: Any):
        self.bot = bot
        self._action_handlers: Dict[str, Callable[[], None]] = {}
        self._last_pressed_button: Optional[ttk.Button] = None

        # Chart histories (kept lightweight)
        self._wallet_history: List[float] = []
        self._virtual_history: List[float] = []
        self._rewards_history: List[float] = []
        self._max_points_seen: float = 0.0

        # try-import matplotlib once
        self._mpl_ok = True
        try:
            from matplotlib.figure import Figure  # noqa: F401
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: F401
        except Exception:
            self._mpl_ok = False

        self.root = tk.Tk()
        self.root.title("AI Trading Terminal")
        self.root.geometry("1700x980")
        self._configure_style()

        self._create_status_bar()
        self._create_left_controls()
        self._create_center_charts()   # <-- three charts
        self._create_right_metrics()
        self._create_bottom_logs()

        self.root.after(UI_REFRESH_INTERVAL, self._refresh)

    # ─────────────────────────────
    # Styling
    # ─────────────────────────────
    def _configure_style(self):
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except Exception:
            pass
        bg = '#1e1e1e'
        style.configure('.', background=bg, foreground='white')
        style.configure('TLabel', background=bg, foreground='white')
        style.configure('TFrame', background=bg)
        style.configure('TLabelframe', background=bg, foreground='white')
        style.configure('TLabelframe.Label', background=bg, foreground='white')
        style.configure('TButton', background='#2a2a2a', foreground='white')

        # Button styles for pressed/active feedback
        style.configure('Primary.TButton', background='#2a2a2a', foreground='white')
        style.configure('Active.TButton', background='#1b5e20', foreground='white')  # green-ish
        style.map('Active.TButton',
                  background=[('active', '#2e7d32'), ('!active', '#1b5e20')])

    # ─────────────────────────────
    # Status Bar
    # ─────────────────────────────
    def _create_status_bar(self):
        frame = ttk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)

        self.conn_label = ttk.Label(frame, text="⚡ Disconnected")
        self.conn_label.pack(side=tk.LEFT, padx=10)

        self.mode_label = ttk.Label(frame, text="Mode: Idle")
        self.mode_label.pack(side=tk.LEFT, padx=10)

        self.heartbeat_label = ttk.Label(frame, text="Heartbeat: --")
        self.heartbeat_label.pack(side=tk.LEFT, padx=10)

        self.last_action_var = tk.StringVar(value="Last action: —")
        self.last_action_label = ttk.Label(frame, textvariable=self.last_action_var)
        self.last_action_label.pack(side=tk.LEFT, padx=20)

        self.time_label = ttk.Label(frame, text=time.strftime('%H:%M:%S'))
        self.time_label.pack(side=tk.RIGHT, padx=10)

    # ─────────────────────────────
    # Left Controls
    # ─────────────────────────────
    def _create_left_controls(self):
        frame = ttk.Labelframe(self.root, text="Controls", padding=10)
        frame.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)

        self.btn_start_train = ttk.Button(
            frame, text="▶ Start Training", style='Primary.TButton',
            command=lambda: self._press_and_invoke(self.btn_start_train, "start_training", "Start Training")
        )
        self.btn_start_train.pack(fill=tk.X, pady=4)

        self.btn_stop_train = ttk.Button(
            frame, text="⏹ Stop Training", style='Primary.TButton',
            command=lambda: self._press_and_invoke(self.btn_stop_train, "stop_training", "Stop Training")
        )
        self.btn_stop_train.pack(fill=tk.X, pady=4)

        ttk.Separator(frame, orient='horizontal').pack(fill=tk.X, pady=6)

        self.btn_start_trade = ttk.Button(
            frame, text="▶ Start Trading", style='Primary.TButton',
            command=lambda: self._press_and_invoke(self.btn_start_trade, "start_trading", "Start Trading")
        )
        self.btn_start_trade.pack(fill=tk.X, pady=8)

        self.btn_stop_trade = ttk.Button(
            frame, text="⏹ Stop Trading", style='Primary.TButton',
            command=lambda: self._press_and_invoke(self.btn_stop_trade, "stop_trading", "Stop Trading")
        )
        self.btn_stop_trade.pack(fill=tk.X, pady=4)

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

        ttk.Button(notif, text="Apply", style='Primary.TButton',
                   command=self._apply_notification_prefs).pack(fill=tk.X, pady=(8, 0))

    def _press_and_invoke(self, btn: ttk.Button, action: str, label: str):
        # Reset old
        if self._last_pressed_button is not None:
            try:
                self._last_pressed_button.configure(style='Primary.TButton')
            except Exception:
                pass
        # Mark new
        try:
            btn.configure(style='Active.TButton')
        except Exception:
            pass
        self._last_pressed_button = btn
        self.last_action_var.set(f"Last action: {label}")
        self._invoke(action)

    # ─────────────────────────────
    # Center Charts (3)
    # ─────────────────────────────
    def _create_center_charts(self):
        if not self._mpl_ok:
            frame = ttk.Labelframe(self.root, text="Charts (matplotlib not available)", padding=10)
            frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)
            self._chart1 = self._chart2 = self._chart3 = None
            return

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        frame = ttk.Labelframe(self.root, text="Charts")
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        # three subframes to avoid one giant subplot; cleaner updates
        self._fig1 = Figure(figsize=(6, 3), dpi=100)
        self._ax1 = self._fig1.add_subplot(111)
        self._ax1.grid(True, alpha=0.2)
        self._canvas1 = FigureCanvasTkAgg(self._fig1, master=frame)
        self._canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._fig2 = Figure(figsize=(6, 3), dpi=100)
        self._ax2 = self._fig2.add_subplot(111)
        self._ax2.grid(True, alpha=0.2)
        self._canvas2 = FigureCanvasTkAgg(self._fig2, master=frame)
        self._canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._fig3 = Figure(figsize=(6, 3), dpi=100)
        self._ax3 = self._fig3.add_subplot(111)
        self._ax3.grid(True, alpha=0.2)
        self._canvas3 = FigureCanvasTkAgg(self._fig3, master=frame)
        self._canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # initial titles
        self._ax1.set_title("Wallet Balance (Live/Paper)")
        self._ax2.set_title("Virtual Wallet Balance (Simulation)")
        self._ax3.set_title("Reward Points")

    # ─────────────────────────────
    # Right Metrics
    # ─────────────────────────────
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

        ttk.Label(frame, text="Reward Points:").pack(anchor=tk.W, pady=(10, 0))
        self.points_var = tk.StringVar(value="0.0")
        ttk.Label(frame, textvariable=self.points_var).pack(anchor=tk.W, pady=2)

    # ─────────────────────────────
    # Bottom Logs
    # ─────────────────────────────
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
        Also feeds chart histories.
        """
        if balance is None:
            balance = kwargs.get("final_balance", 0.0)
        if points is None:
            points = kwargs.get("total_points", 0.0)

        # Update textual
        self.balance_var.set(f"${balance:,.2f}")
        self.points_var.set(f"{points:,.2f}")

        # Push into histories
        self._append_histories(balance=balance, virtual_balance=balance, points=points)

        self.log(f"Simulation: Balance=${balance:,.2f}, Points={points:.2f}", level='SUCCESS')

    def update_live_metrics(self, metrics: Dict[str, Any]):
        bal = metrics.get("balance")
        if bal is not None:
            self.balance_var.set(f"${bal:,.2f}")
            self._append_histories(balance=float(bal))

        eq = metrics.get("equity")
        if eq is not None:
            self.value_var.set(f"${eq:,.2f}")

        sym = metrics.get("symbol")
        if sym:
            self.symbol_var.set(sym)

        tf = metrics.get("timeframe")
        if tf:
            self.tf_var.set(tf)

        points = metrics.get("points")
        if points is not None:
            self.points_var.set(f"{float(points):,.2f}")
            self._append_histories(points=float(points))

        vbal = metrics.get("virtual_balance")
        if vbal is not None:
            self._append_histories(virtual_balance=float(vbal))

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

    def _append_histories(self, *, balance: float = None, virtual_balance: float = None, points: float = None):
        # keep last 500 points
        maxlen = 500
        if balance is not None:
            self._wallet_history.append(float(balance))
            if len(self._wallet_history) > maxlen:
                self._wallet_history = self._wallet_history[-maxlen:]
        if virtual_balance is not None:
            self._virtual_history.append(float(virtual_balance))
            if len(self._virtual_history) > maxlen:
                self._virtual_history = self._virtual_history[-maxlen:]
        if points is not None:
            self._rewards_history.append(float(points))
            if float(points) > self._max_points_seen:
                self._max_points_seen = float(points)
            if len(self._rewards_history) > maxlen:
                self._rewards_history = self._rewards_history[-maxlen:]

    def _refresh(self):
        # status clocks and basic labels
        self.time_label.config(text=time.strftime('%H:%M:%S'))

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

        # live pull-through (optional)
        bal = getattr(self.bot, "current_balance", None)
        if bal is not None:
            self.balance_var.set(f"${bal:,.2f}")
            self._append_histories(balance=float(bal))

        vbal = getattr(self.bot, "virtual_balance", None)
        if vbal is not None:
            self._append_histories(virtual_balance=float(vbal))

        pts = getattr(self.bot, "reward_points", None)
        if pts is not None:
            self.points_var.set(f"{float(pts):,.2f}")
            self._append_histories(points=float(pts))

        sym = getattr(self.bot, "current_symbol", None)
        if sym:
            self.symbol_var.set(sym)

        tf = getattr(self.bot, "timeframe", None)
        if tf:
            self.tf_var.set(tf)

        # redraw charts if matplotlib is available
        if self._mpl_ok:
            try:
                if len(self._wallet_history) > 1:
                    self._ax1.clear()
                    self._ax1.plot(range(len(self._wallet_history)), self._wallet_history)
                    self._ax1.grid(True, alpha=0.2)
                    self._ax1.set_title("Wallet Balance (Live/Paper)")
                    self._canvas1.draw()

                if len(self._virtual_history) > 1:
                    self._ax2.clear()
                    self._ax2.plot(range(len(self._virtual_history)), self._virtual_history)
                    self._ax2.grid(True, alpha=0.2)
                    self._ax2.set_title("Virtual Wallet Balance (Simulation)")
                    self._canvas2.draw()

                if len(self._rewards_history) > 1:
                    self._ax3.clear()
                    self._ax3.plot(range(len(self._rewards_history)), self._rewards_history)
                    self._ax3.grid(True, alpha=0.2)
                    self._ax3.set_title("Reward Points")
                    self._canvas3.draw()
            except Exception as e:
                self.log(f"Chart update failed: {e}", level='ERROR')

        self.root.after(UI_REFRESH_INTERVAL, self._refresh)

# modules/ui.py

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import time
from typing import Callable, Dict, Any, Optional, List, Tuple

import config

UI_REFRESH_INTERVAL = getattr(config, "UI_REFRESH_INTERVAL", 1000)


class TradingUI:
    """
    GUI for controlling and monitoring the TradingBot.

    Changes:
      - Three matplotlib charts: Wallet Balance, Virtual Portfolio Value, Reward Points
      - Buttons change color when pressed (to show last action)
      - Simple series appenders to update charts efficiently
    """

    def __init__(self, bot: Any):
        self.bot = bot
        self._action_handlers: Dict[str, Callable[[], None]] = {}
        self._last_pressed_btn: Optional[ttk.Button] = None
        self._all_buttons: List[ttk.Button] = []

        self._wallet_points: List[Tuple[float, float]] = []     # (epoch_s, value)
        self._virtual_points: List[Tuple[float, float]] = []    # (epoch_s, value)
        self._reward_points: List[Tuple[float, float]] = []     # (epoch_s, value)

        self.root = tk.Tk()
        self.root.title("AI Trading Terminal")
        self.root.geometry("1600x900")
        self._configure_style()

        self._create_status_bar()
        self._create_left_controls()
        self._create_center_charts()  # triple charts
        self._create_right_metrics()
        self._create_bottom_logs()

        self.root.after(UI_REFRESH_INTERVAL, self._refresh)

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
                  foreground=[('disabled', '#888888')])

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

    def _btn(self, parent, text, action) -> ttk.Button:
        b = ttk.Button(parent, text=text, command=lambda: self._on_button(action))
        self._all_buttons.append(b)
        return b

    def _on_button(self, action: str):
        # visual feedback
        for b in self._all_buttons:
            b.configure(style='TButton')
        caller = None
        try:
            # find the widget that triggered this call (simplest: set color of last pressed)
            # ttk doesn't pass the widget automatically, so we color all same and then color the focused one
            caller = self.root.focus_displayof()
        except Exception:
            pass
        # color the *intended* button by scanning by text (robust enough for this UI)
        for b in self._all_buttons:
            if b.cget("text").lower().find(action.split("_")[1]) >= 0:
                # create a custom style on the fly
                hot = ttk.Style()
                hot.configure('Hot.TButton', background='#0059b3', foreground='white')
                b.configure(style='Hot.TButton')
                self._last_pressed_btn = b
                break

        # invoke
        self._invoke(action)

    def _create_left_controls(self):
        frame = ttk.Labelframe(self.root, text="Controls", padding=10)
        frame.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)

        self._btn(frame, "▶ Start Training", "start_training").pack(fill=tk.X, pady=4)
        self._btn(frame, "⏹ Stop Training", "stop_training").pack(fill=tk.X, pady=4)

        ttk.Separator(frame, orient='horizontal').pack(fill=tk.X, pady=6)

        self._btn(frame, "▶ Start Trading", "start_trading").pack(fill=tk.X, pady=8)
        self._btn(frame, "⏹ Stop Trading", "stop_trading").pack(fill=tk.X, pady=4)

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
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except Exception:
            self.wallet_ax = None
            self.virtual_ax = None
            self.rewards_ax = None
            frame = ttk.Labelframe(self.root, text="Charts (matplotlib not available)", padding=10)
            frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)
            return

        frame = ttk.Labelframe(self.root, text="Performance Charts")
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Wallet Balance
        fig1 = Figure(figsize=(5, 3), dpi=100)
        self.wallet_ax = fig1.add_subplot(111)
        self.wallet_ax.grid(True, alpha=0.2)
        self.wallet_canvas = FigureCanvasTkAgg(fig1, master=frame)
        self.wallet_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Virtual Portfolio Value
        fig2 = Figure(figsize=(5, 3), dpi=100)
        self.virtual_ax = fig2.add_subplot(111)
        self.virtual_ax.grid(True, alpha=0.2)
        self.virtual_canvas = FigureCanvasTkAgg(fig2, master=frame)
        self.virtual_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Reward Points
        fig3 = Figure(figsize=(5, 3), dpi=100)
        self.rewards_ax = fig3.add_subplot(111)
        self.rewards_ax.grid(True, alpha=0.2)
        self.rewards_canvas = FigureCanvasTkAgg(fig3, master=frame)
        self.rewards_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

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
        self.tf_var = tk.StringVar(value="5m / 15m")
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
        Also appends the points to chart series.
        """
        if balance is None:
            balance = kwargs.get("final_balance", 0.0)
        if points is None:
            points = kwargs.get("total_points", 0.0)

        self.balance_var.set(f"${balance:,.2f}")
        self.append_wallet_point(balance)
        self.append_reward_point(points)
        self.log(f"Simulation: Balance=${balance:,.2f}, Points={points:.2f}", level='SUCCESS')

    def update_live_metrics(self, metrics: Dict[str, Any]):
        bal = metrics.get("balance")
        if bal is not None:
            self.balance_var.set(f"${bal:,.2f}")
            self.append_wallet_point(float(bal))
        eq = metrics.get("equity")
        if eq is not None:
            self.value_var.set(f"${eq:,.2f}")
            self.append_virtual_point(float(eq))
        sym = metrics.get("symbol")
        if sym:
            self.symbol_var.set(sym)
        tf = metrics.get("timeframe")
        if tf:
            self.tf_var.set(tf)

    def append_wallet_point(self, value: float):
        self._wallet_points.append((time.time(), value))
        self._wallet_points = self._wallet_points[-500:]  # keep last N for UI

    def append_virtual_point(self, value: float):
        self._virtual_points.append((time.time(), value))
        self._virtual_points = self._virtual_points[-500:]

    def append_reward_point(self, total_points: float):
        self._reward_points.append((time.time(), total_points))
        self._reward_points = self._reward_points[-500:]

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

        # redraw charts
        try:
            if self.wallet_ax:
                self.wallet_ax.clear()
                self.wallet_ax.grid(True, alpha=0.2)
                if self._wallet_points:
                    xs = [p[0] for p in self._wallet_points]
                    ys = [p[1] for p in self._wallet_points]
                    self.wallet_ax.plot(xs, ys)
                self.wallet_ax.set_title("Wallet Balance")
                self.wallet_canvas.draw()

            if self.virtual_ax:
                self.virtual_ax.clear()
                self.virtual_ax.grid(True, alpha=0.2)
                if self._virtual_points:
                    xs = [p[0] for p in self._virtual_points]
                    ys = [p[1] for p in self._virtual_points]
                    self.virtual_ax.plot(xs, ys)
                self.virtual_ax.set_title("Virtual Portfolio Value")
                self.virtual_canvas.draw()

            if self.rewards_ax:
                self.rewards_ax.clear()
                self.rewards_ax.grid(True, alpha=0.2)
                if self._reward_points:
                    xs = [p[0] for p in self._reward_points]
                    ys = [p[1] for p in self._reward_points]
                    self.rewards_ax.plot(xs, ys)
                self.rewards_ax.set_title("Reward Points")
                self.rewards_canvas.draw()
        except Exception as e:
            self.log(f"Chart update failed: {e}", level='ERROR')

        self.root.after(UI_REFRESH_INTERVAL, self._refresh)

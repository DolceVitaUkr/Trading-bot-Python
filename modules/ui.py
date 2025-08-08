# modules/ui.py

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import time
from typing import Callable, Dict, Any, Optional

import config

UI_REFRESH_INTERVAL = getattr(config, "UI_REFRESH_INTERVAL", 1000)


class TradingUI:
    """
    GUI for controlling and monitoring the TradingBot.
    """

    def __init__(self, bot: Any):
        self.bot = bot
        self._action_handlers: Dict[str, Callable[[], None]] = {}

        self.root = tk.Tk()
        self.root.title("AI Trading Terminal")
        self.root.geometry("1600x900")
        self._configure_style()

        self._create_status_bar()
        self._create_left_controls()
        self._create_center_chart()
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
        style.configure('TNotebook', background='#2d2d2d')
        style.configure('TNotebook.Tab', background='#2d2d2d', foreground='white')
        style.map('TNotebook.Tab', background=[('selected', '#3e3e3e')])

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

        ttk.Button(frame, text="▶ Start Training", command=lambda: self._invoke("start_training")).pack(fill=tk.X, pady=4)
        ttk.Button(frame, text="⏹ Stop Training", command=lambda: self._invoke("stop_training")).pack(fill=tk.X, pady=4)

        ttk.Separator(frame, orient='horizontal').pack(fill=tk.X, pady=6)

        ttk.Button(frame, text="▶ Start Trading", command=lambda: self._invoke("start_trading")).pack(fill=tk.X, pady=8)
        ttk.Button(frame, text="⏹ Stop Trading", command=lambda: self._invoke("stop_trading")).pack(fill=tk.X, pady=4)

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

    def _create_center_chart(self):
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except Exception:
            self.ax = None
            self.chart_canvas = None
            frame = ttk.Labelframe(self.root, text="Price Chart (matplotlib not available)", padding=10)
            frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)
            return

        frame = ttk.Labelframe(self.root, text="Price Chart")
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        fig = Figure(figsize=(8, 5), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.grid(True, alpha=0.2)
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

        bal = getattr(self.bot, "current_balance", None)
        if bal is not None:
            self.balance_var.set(f"${bal:,.2f}")

        val = getattr(self.bot, "portfolio_value", None)
        if val is not None:
            self.value_var.set(f"${val:,.2f}")

        sym = getattr(self.bot, "current_symbol", None)
        if sym:
            self.symbol_var.set(sym)

        tf = getattr(self.bot, "timeframe", None)
        if tf:
            self.tf_var.set(tf)

        try:
            dm = getattr(self.bot, "data_manager", None)
            sym = getattr(self.bot, "current_symbol", None)
            tf = getattr(self.bot, "timeframe", None)
            if dm and hasattr(dm, "load_historical_data") and sym and tf and getattr(self, "ax", None):
                df = dm.load_historical_data(sym, tf)
                if len(df) > 0:
                    self.ax.clear()
                    self.ax.plot(df.index, df["close"])
                    self.ax.set_title(f"{sym} ({tf})")
                    self.ax.grid(True, alpha=0.2)
                    self.chart_canvas.draw()
        except Exception as e:
            self.log(f"Chart update failed: {e}", level='ERROR')

        self.root.after(UI_REFRESH_INTERVAL, self._refresh)



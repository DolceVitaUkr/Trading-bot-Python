# modules/ui.py

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Callable, Dict, Any

import config
from main import TradingBot  # assume TradingBot is defined in main.py

# Safe default for UI refresh interval (ms)
UI_REFRESH_INTERVAL = getattr(config, "UI_REFRESH_INTERVAL", 1000)


class TradingUI:
    """
    GUI for controlling and monitoring the TradingBot.
    - Accepts a TradingBot instance to drive core logic.
    - Provides buttons for Start/Stop Training and Start/Stop Trading.
    - Displays performance metrics, logs, and real-time charts.
    """

    def __init__(self, bot: TradingBot):
        self.bot = bot
        self._action_handlers: Dict[str, Callable[[], None]] = {}

        # Initialize main window
        self.root = tk.Tk()
        self.root.title("AI Trading Terminal v2.0")
        self.root.geometry("1600x900")
        self._configure_style()

        # Build UI
        self._create_status_bar()
        self._create_control_panel()
        self._create_chart_panel()
        self._create_metrics_panel()
        self._create_log_panel()

        # Schedule periodic refresh
        self.root.after(UI_REFRESH_INTERVAL, self._refresh)

    def _configure_style(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', background='#1e1e1e', foreground='white')
        style.configure('TNotebook', background='#2d2d2d')
        style.configure('TNotebook.Tab', background='#2d2d2d', foreground='white')
        style.map('TNotebook.Tab', background=[('selected', '#3e3e3e')])

    def _create_status_bar(self):
        frame = ttk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.X)
        self.conn_label = ttk.Label(frame, text="⚡ Disconnected")
        self.conn_label.pack(side=tk.LEFT, padx=10)
        self.mode_label = ttk.Label(frame, text="Mode: N/A")
        self.mode_label.pack(side=tk.LEFT, padx=10)
        self.time_label = ttk.Label(frame, text=time.strftime('%H:%M:%S'))
        self.time_label.pack(side=tk.RIGHT, padx=10)

    def _create_control_panel(self):
        frame = ttk.LabelFrame(self.root, text="Controls", padding=10)
        frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        btn_start_train = ttk.Button(frame, text="▶ Start Training", command=lambda: self._invoke("start_training"))
        btn_start_train.pack(fill=tk.X, pady=2)
        btn_stop_train = ttk.Button(frame, text="⏹ Stop Training", command=lambda: self._invoke("stop_training"))
        btn_stop_train.pack(fill=tk.X, pady=2)

        btn_start_live = ttk.Button(frame, text="▶ Start Trading", command=lambda: self._invoke("start_trading"))
        btn_start_live.pack(fill=tk.X, pady=10)
        btn_stop_live = ttk.Button(frame, text="⏹ Stop Trading", command=lambda: self._invoke("stop_trading"))
        btn_stop_live.pack(fill=tk.X, pady=2)

    def _create_chart_panel(self):
        frame = ttk.LabelFrame(self.root, text="Price Chart")
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        fig = Figure(figsize=(8, 5), dpi=100)
        self.ax = fig.add_subplot(111)
        self.chart_canvas = FigureCanvasTkAgg(fig, master=frame)
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _create_metrics_panel(self):
        frame = ttk.LabelFrame(self.root, text="Metrics", padding=10)
        frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        ttk.Label(frame, text="Wallet Balance:").pack(anchor=tk.W)
        self.balance_var = tk.StringVar(value="$0.00")
        ttk.Label(frame, textvariable=self.balance_var).pack(anchor=tk.W, pady=2)

        ttk.Label(frame, text="Simulation Points:").pack(anchor=tk.W, pady=(10,0))
        self.points_var = tk.StringVar(value="0")
        ttk.Label(frame, textvariable=self.points_var).pack(anchor=tk.W, pady=2)

        ttk.Label(frame, text="Portfolio Value:").pack(anchor=tk.W, pady=(10,0))
        self.value_var = tk.StringVar(value="$0.00")
        ttk.Label(frame, textvariable=self.value_var).pack(anchor=tk.W, pady=2)

    def _create_log_panel(self):
        frame = ttk.LabelFrame(self.root, text="Logs")
        frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        self.log_text = scrolledtext.ScrolledText(frame, height=8, bg='#2d2d2d', fg='white')
        self.log_text.pack(fill=tk.BOTH, expand=True)
        # configure tags
        self.log_text.tag_config('INFO', foreground='#00ff99')
        self.log_text.tag_config('ERROR', foreground='#ff3300')
        self.log_text.tag_config('SUCCESS', foreground='#00ccff')

    def add_action_handler(self, name: str, callback: Callable[[], None]):
        """
        Register a callback for a named UI action.
        Supported names: 'start_training', 'stop_training', 'start_trading', 'stop_trading'
        """
        self._action_handlers[name] = callback

    def update_simulation_results(self, balance: float, points: float):
        """
        Display final simulation results.
        """
        self.balance_var.set(f"${balance:,.2f}")
        self.points_var.set(f"{points:.2f}")
        self.log(f"Simulation complete: Balance=${balance:,.2f}, Points={points:.2f}", level='SUCCESS')

    def log(self, message: str, level: str = 'INFO'):
        """
        Append a log message to the log panel.
        """
        ts = time.strftime('%H:%M:%S')
        self.log_text.insert(tk.END, f"{ts} - {message}\n", level)
        self.log_text.see(tk.END)

    def _invoke(self, action: str):
        """
        Invoke a registered action handler.
        """
        cb = self._action_handlers.get(action)
        if not cb:
            messagebox.showwarning("Not implemented", f"No handler for '{action}'")
            return
        try:
            cb()
        except Exception as e:
            self.log(f"Error in action '{action}': {e}", level='ERROR')

    def _refresh(self):
        """
        Periodic UI refresh: update time, chart, and live metrics.
        """
        # Update time
        self.time_label.config(text=time.strftime('%H:%M:%S'))

        # Update connection/mode status
        conn = "Connected" if self.bot.is_connected else "Disconnected"
        self.conn_label.config(text=f"⚡ {conn}")
        mode = "Training" if self.bot.is_training else "Live" if self.bot.is_trading else "Idle"
        self.mode_label.config(text=f"Mode: {mode}")

        # Update balance & portfolio value
        bal = self.bot.current_balance
        val = self.bot.portfolio_value
        self.balance_var.set(f"${bal:,.2f}")
        self.value_var.set(f"${val:,.2f}")

        # Update chart with latest data
        try:
            df = self.bot.data_manager.load_historical_data(self.bot.current_symbol, self.bot.timeframe)
            self.ax.clear()
            self.ax.plot(df['close'])
            self.chart_canvas.draw()
        except Exception as e:
            self.log(f"Chart update failed: {e}", level='ERROR')

        # Schedule next refresh
        self.root.after(UI_REFRESH_INTERVAL, self._refresh)

    def run(self):
        """
        Start the Tkinter main loop.
        """
        self.root.mainloop()

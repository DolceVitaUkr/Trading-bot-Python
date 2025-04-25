import tkinter as tk
from tkinter import ttk, messagebox
import threading
import logging
import time
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from modules.exchange import ExchangeAPI
from modules.trade_simulator import TradeSimulator
from modules.data_manager import DataManager
from modules.error_handler import ErrorHandler
from self_test.full_system_test import SystemTestRunner
from config import UI_REFRESH_INTERVAL

class TradingUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI Trading Terminal v2.0")
        self.root.geometry("1600x900")
        self._configure_style()
        
        self.exchange = ExchangeAPI()
        self.data_manager = DataManager()
        self.error_handler = ErrorHandler()
        self.running = False
        self.update_interval = UI_REFRESH_INTERVAL
        
        self._create_status_bar()
        self._create_left_panel()
        self._create_center_panel()
        self._create_right_panel()
        self._create_bottom_panel()
        
        self.root.after(100, self._update_data)

    def _configure_style(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', background='#1e1e1e', foreground='white')
        style.configure('TNotebook', background='#2d2d2d')
        style.configure('TNotebook.Tab', background='#2d2d2d', foreground='white')
        style.map('TNotebook.Tab', background=[('selected', '#3e3e3e')])

    def _create_status_bar(self):
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.connection_status = ttk.Label(status_frame, text="🟢 Connected")
        self.connection_status.pack(side=tk.LEFT, padx=10)
        
        self.test_status = ttk.Label(status_frame, text="")
        self.test_status.pack(side=tk.LEFT, padx=10)
        
        self.balance_status = ttk.Label(status_frame, text="Balance: $10,000.00")
        self.balance_status.pack(side=tk.RIGHT, padx=10)
        
        self.time_label = ttk.Label(status_frame, text=time.strftime('%H:%M:%S'))
        self.time_label.pack(side=tk.RIGHT, padx=10)

    def _create_left_panel(self):
        left_frame = ttk.Frame(self.root, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        control_frame = ttk.LabelFrame(left_frame, text="Control Panel")
        control_frame.pack(pady=10, padx=5, fill=tk.X)
        
        ttk.Button(control_frame, text="▶ Start Bot", command=self.start_bot).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="⏹ Stop Bot", command=self.stop_bot).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="⚙ Run Self-Test", command=self.run_self_test).pack(fill=tk.X, pady=2)
        
        ttk.Label(left_frame, text="Trading Mode").pack(anchor=tk.W)
        self.mode_var = tk.StringVar(value='paper')
        ttk.Radiobutton(left_frame, text="📝 Paper Trading", variable=self.mode_var, value='paper').pack(anchor=tk.W)
        ttk.Radiobutton(left_frame, text="💵 Live Trading", variable=self.mode_var, value='live').pack(anchor=tk.W)
        
        account_frame = ttk.LabelFrame(left_frame, text="Account")
        account_frame.pack(pady=10, fill=tk.X)
        ttk.Label(account_frame, text="Equity:").pack(anchor=tk.W)
        self.equity_label = ttk.Label(account_frame, text="$10,000.00")
        self.equity_label.pack(anchor=tk.W)

    def _create_center_panel(self):
        center_frame = ttk.Frame(self.root)
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        notebook = ttk.Notebook(center_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        fig = Figure(figsize=(8, 5), dpi=100)
        self.ax = fig.add_subplot(111)
        self.chart = FigureCanvasTkAgg(fig, notebook)
        notebook.add(self.chart.get_tk_widget(), text="Price Chart")

    def _create_right_panel(self):
        right_frame = ttk.Frame(self.root, width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        order_frame = ttk.LabelFrame(right_frame, text="Order Entry")
        order_frame.pack(pady=10, fill=tk.X)
        
        ttk.Label(order_frame, text="Symbol").pack(anchor=tk.W)
        self.symbol_entry = ttk.Combobox(order_frame, values=["BTC/USDT"])
        self.symbol_entry.pack(fill=tk.X)
        
        ttk.Button(order_frame, text="Buy", command=lambda: self.place_order('buy')).pack(side=tk.LEFT, padx=2)
        ttk.Button(order_frame, text="Sell", command=lambda: self.place_order('sell')).pack(side=tk.RIGHT, padx=2)
        
        positions_frame = ttk.LabelFrame(right_frame, text="Positions")
        positions_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        self.positions_tree = ttk.Treeview(positions_frame, columns=('Symbol', 'Size', 'P/L'))
        self.positions_tree.pack(fill=tk.BOTH, expand=True)

    def _create_bottom_panel(self):
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        log_frame = ttk.LabelFrame(bottom_frame, text="Logs")
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.scrolledtext.ScrolledText(log_frame, height=10, bg='#2d2d2d', fg='white')
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _update_data(self):
        try:
            df = self.data_manager.load_historical_data("BTC/USDT")
            self.ax.clear()
            self.ax.plot(df['close'], color='#00ff99')
            self.chart.draw()
            self.time_label.config(text=time.strftime('%H:%M:%S'))
        except Exception as e:
            self.log(f"Error updating data: {str(e)}", 'error')
        self.root.after(self.update_interval, self._update_data)

    def run_self_test(self):
        def run_tests():
            try:
                self.log("Starting system self-test...", 'info')
                tester = SystemTestRunner()
                tester.run_full_suite()
                self.log("Self-test completed successfully", 'success')
            except Exception as e:
                self.log(f"Self-test failed: {str(e)}", 'error')
        
        threading.Thread(target=run_tests, daemon=True).start()

    def log(self, message: str, level: str = 'info'):
        color = {
            'info': '#00ff99',
            'error': '#ff3300',
            'success': '#00cc00'
        }.get(level, 'white')
        
        self.log_text.tag_config(level, foreground=color)
        self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n", level)
        self.log_text.see(tk.END)

    def start_bot(self):
        self.running = True
        self.connection_status.config(text="🟢 Running")
        threading.Thread(target=self._run_strategy, daemon=True).start()

    def stop_bot(self):
        self.running = False
        self.connection_status.config(text="🔴 Stopped")

    def place_order(self, side: str):
        order = {
            'symbol': self.symbol_entry.get(),
            'side': side,
            'amount': 0.001,
            'price': 50000
        }
        try:
            self.exchange.create_order(**order)
            self.log(f"{side.capitalize()} order executed", 'info')
        except Exception as e:
            self.log(f"Order failed: {str(e)}", 'error')

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    ui = TradingUI()
    ui.run()
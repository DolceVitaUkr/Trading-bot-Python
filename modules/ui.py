import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from modules.exchange import ExchangeAPI
from modules.trade_simulator import TradeSimulator
from modules.data_manager import DataManager
from modules.error_handler import ErrorHandler

# Safe default for UI refresh interval (ms) if not defined in config
try:
    from config import UI_REFRESH_INTERVAL
except ImportError:
    UI_REFRESH_INTERVAL = 1000  # default to 1 second

class TradingUI:
    def __init__(self):
        # Initialize main window
        self.root = tk.Tk()
        self.root.title("AI Trading Terminal v2.0")
        self.root.geometry("1600x900")
        self._configure_style()

        # Initialize core components
        self.exchange = ExchangeAPI()
        self.data_manager = DataManager()
        self.error_handler = ErrorHandler()

        self.running = False
        self.update_interval = UI_REFRESH_INTERVAL

        # Build UI panels
        self._create_status_bar()
        self._create_left_panel()
        self._create_center_panel()
        self._create_right_panel()
        self._create_bottom_panel()

        # Start periodic update loop
        self.root.after(100, self._update_data)

    def _configure_style(self):
        """Configure the application style and theme."""
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', background='#1e1e1e', foreground='white')
        style.configure('TNotebook', background='#2d2d2d')
        style.configure('TNotebook.Tab', background='#2d2d2d', foreground='white')
        style.map('TNotebook.Tab', background=[('selected', '#3e3e3e')])

    def _create_status_bar(self):
        """Create the top status bar with connection and time."""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.TOP, fill=tk.X)

        self.connection_status = ttk.Label(status_frame, text=" Connected")
        self.connection_status.pack(side=tk.LEFT, padx=10)

        self.test_status = ttk.Label(status_frame, text="")
        self.test_status.pack(side=tk.LEFT, padx=10)

        self.balance_status = ttk.Label(status_frame, text="Balance: $10,000.00")
        self.balance_status.pack(side=tk.RIGHT, padx=10)

        self.time_label = ttk.Label(status_frame, text=time.strftime('%H:%M:%S'))
        self.time_label.pack(side=tk.RIGHT, padx=10)

    def _create_left_panel(self):
        """Create the left control panel with start/stop and mode selection."""
        left_frame = ttk.Frame(self.root, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        control_frame = ttk.LabelFrame(left_frame, text="Control Panel")
        control_frame.pack(pady=10, padx=5, fill=tk.X)

        ttk.Button(control_frame, text="▶ Start Bot", command=self.start_bot).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="⏹ Stop Bot", command=self.stop_bot).pack(fill=tk.X, pady=2)
        # Removed "Run Self-Test" button to separate test logic from UI

        ttk.Label(left_frame, text="Trading Mode").pack(anchor=tk.W)
        self.mode_var = tk.StringVar(value='paper')
        ttk.Radiobutton(left_frame, text=" Paper Trading", variable=self.mode_var, value='paper').pack(anchor=tk.W)
        ttk.Radiobutton(left_frame, text=" Live Trading", variable=self.mode_var, value='live').pack(anchor=tk.W)

        account_frame = ttk.LabelFrame(left_frame, text="Account")
        account_frame.pack(pady=10, fill=tk.X)
        ttk.Label(account_frame, text="Equity:").pack(anchor=tk.W)
        self.equity_label = ttk.Label(account_frame, text="$10,000.00")
        self.equity_label.pack(anchor=tk.W)

    def _create_center_panel(self):
        """Create the central panel with a notebook for charts or other views."""
        center_frame = ttk.Frame(self.root)
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        notebook = ttk.Notebook(center_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Price Chart tab
        fig = Figure(figsize=(8, 5), dpi=100)
        self.ax = fig.add_subplot(111)
        self.chart = FigureCanvasTkAgg(fig, notebook)
        notebook.add(self.chart.get_tk_widget(), text="Price Chart")

        # (Additional tabs for metrics or depth charts can be added to the notebook)

    def _create_right_panel(self):
        """Create the right panel for order entry and positions display."""
        right_frame = ttk.Frame(self.root, width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Order Entry section
        order_frame = ttk.LabelFrame(right_frame, text="Order Entry")
        order_frame.pack(pady=10, fill=tk.X)
        ttk.Label(order_frame, text="Symbol").pack(anchor=tk.W)
        self.symbol_entry = ttk.Combobox(order_frame, values=["BTC/USDT"])
        self.symbol_entry.pack(fill=tk.X)
        ttk.Button(order_frame, text="Buy", command=lambda: self.place_order('buy')).pack(side=tk.LEFT, padx=2)
        ttk.Button(order_frame, text="Sell", command=lambda: self.place_order('sell')).pack(side=tk.RIGHT, padx=2)

        # Positions section
        positions_frame = ttk.LabelFrame(right_frame, text="Positions")
        positions_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        self.positions_tree = ttk.Treeview(positions_frame, columns=('Symbol', 'Size', 'P/L'))
        self.positions_tree.pack(fill=tk.BOTH, expand=True)

    def _create_bottom_panel(self):
        """Create the bottom panel for logs."""
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        log_frame = ttk.LabelFrame(bottom_frame, text="Logs")
        log_frame.pack(fill=tk.BOTH, expand=True)
        # Use ScrolledText for log output
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, bg='#2d2d2d', fg='white')
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _update_data(self):
        """Periodic update of data (price chart, time, etc.)."""
        try:
            # Example: load and plot latest historical data for a symbol
            df = self.data_manager.load_historical_data("BTC/USDT")
            self.ax.clear()
            self.ax.plot(df['close'], color='#00ff99')
            self.chart.draw()

            # Update time in status bar
            self.time_label.config(text=time.strftime('%H:%M:%S'))
        except Exception as e:
            # Log any error in updating data
            self.log(f"Error updating data: {str(e)}", level='error')
        # Schedule the next update
        self.root.after(self.update_interval, self._update_data)

    def log(self, message: str, level: str = 'info'):
        """Log a message to the log text area with a given level (info, error, success)."""
        colors = {
            'info': '#00ff99',
            'error': '#ff3300',
            'success': '#00cc00'
        }
        color = colors.get(level, 'white')
        # Configure tag for this level if not already done
        self.log_text.tag_config(level, foreground=color)
        # Insert timestamped message with the level tag
        timestamp = time.strftime('%H:%M:%S')
        self.log_text.insert(tk.END, f"{timestamp} - {message}\n", level)
        self.log_text.see(tk.END)  # Auto-scroll to the end

    def start_bot(self):
        """Start the trading bot in a background thread."""
        self.running = True
        self.connection_status.config(text=" Running")
        threading.Thread(target=self._run_strategy, daemon=True).start()

    def stop_bot(self):
        """Stop the trading bot."""
        self.running = False
        self.connection_status.config(text=" Stopped")

    def place_order(self, side: str):
        """Place a buy or sell order through the exchange API."""
        order = {
            'symbol': self.symbol_entry.get(),
            'side': side,
            'amount': 0.001,
            'price': 50000  # In a real bot, price would be current market or specified
        }
        try:
            self.exchange.create_order(**order)
            self.log(f"{side.capitalize()} order executed", level='info')
        except Exception as e:
            self.log(f"Order failed: {str(e)}", level='error')

    def _run_strategy(self):
        """Background thread entry point for running the trading strategy loop."""
        # This is a placeholder for the actual trading logic.
        # It would continually run while self.running is True.
        while self.running:
            try:
                # Example strategy step (could call TradeSimulator or ExchangeAPI methods)
                time.sleep(1)  # placeholder for real strategy work
            except Exception as e:
                self.log(f"Strategy error: {str(e)}", level='error')
                self.running = False

    def run(self):
        """Start the Tkinter main loop to display the UI."""
        self.root.mainloop()

# If this module is run directly, launch the UI.
if __name__ == "__main__":
    ui = TradingUI()
    ui.run()

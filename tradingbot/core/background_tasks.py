"""
Background Task Manager for Trading Engine
Handles async tasks in FastAPI context
"""
import asyncio
from typing import Dict, Optional, Set
from concurrent.futures import ThreadPoolExecutor
import threading

from .loggerconfig import get_logger


class BackgroundTaskManager:
    """Manages background tasks for trading operations."""
    
    def __init__(self):
        self.log = get_logger("background_tasks")
        self.tasks: Dict[str, asyncio.Task] = {}
        self.running_threads: Dict[str, threading.Thread] = {}
        self.stop_events: Dict[str, threading.Event] = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    def start_trading_task(self, asset: str, trading_func, *args):
        """Start a trading task in a background thread."""
        task_id = f"trading_{asset}"
        
        if task_id in self.running_threads and self.running_threads[task_id].is_alive():
            self.log.warning(f"Trading task for {asset} already running")
            return False
            
        # Create stop event
        stop_event = threading.Event()
        self.stop_events[task_id] = stop_event
        
        # Create and start thread
        thread = threading.Thread(
            target=self._run_trading_loop,
            args=(asset, trading_func, stop_event, *args),
            name=f"TradingThread-{asset}"
        )
        thread.daemon = True
        thread.start()
        
        self.running_threads[task_id] = thread
        self.log.info(f"Started trading task for {asset}")
        return True
        
    def _run_trading_loop(self, asset: str, trading_func, stop_event: threading.Event, *args):
        """Run trading loop in thread."""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the trading function
            loop.run_until_complete(trading_func(asset, stop_event, *args))
            
        except Exception as e:
            self.log.error(f"Error in trading loop for {asset}: {e}", exc_info=True)
        finally:
            loop.close()
            
    def stop_trading_task(self, asset: str):
        """Stop a trading task."""
        task_id = f"trading_{asset}"
        
        # Set stop event
        if task_id in self.stop_events:
            self.stop_events[task_id].set()
            
        # Wait for thread to finish
        if task_id in self.running_threads:
            thread = self.running_threads[task_id]
            if thread.is_alive():
                thread.join(timeout=5.0)
                
            del self.running_threads[task_id]
            del self.stop_events[task_id]
            
        self.log.info(f"Stopped trading task for {asset}")
        
    def is_running(self, asset: str) -> bool:
        """Check if trading task is running."""
        task_id = f"trading_{asset}"
        return (
            task_id in self.running_threads and 
            self.running_threads[task_id].is_alive()
        )
        
    def stop_all(self):
        """Stop all trading tasks."""
        for asset in list(self.stop_events.keys()):
            self.stop_trading_task(asset.replace("trading_", ""))
            
        self.executor.shutdown(wait=True)
        

# Global instance
background_tasks = BackgroundTaskManager()
"""Application entry point launching the FastAPI dashboard."""

from __future__ import annotations

import sys
import os
import asyncio
import threading
import time
from pathlib import Path

# Add the parent directory to Python path if not already there
current_dir = Path(__file__).parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import uvicorn

from tradingbot.ui.app import app


def run_server():
    """Run the FastAPI server in a separate thread."""
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


def main() -> None:
    """Start the FastAPI server in background and continue with bot operations."""
    # Start FastAPI server in a background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    print("\n[SUCCESS] Dashboard server started on http://localhost:8000")
    print("[INFO] Bot is running. Press Ctrl+C to stop.\n")
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down bot...")
        sys.exit(0)


if __name__ == "__main__":
    main()
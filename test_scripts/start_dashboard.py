#!/usr/bin/env python3
"""
Trading Bot Dashboard Launcher
Starts the FastAPI dashboard server.
"""

import sys
import uvicorn
from pathlib import Path

# Ensure the project root is in Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tradingbot.ui.app import app

def find_available_port(start_port=8000, max_attempts=10):
    """Find an available port starting from start_port."""
    import socket
    
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    return None

def main():
    """Start the trading bot dashboard."""
    print("🚀 Starting Trading Bot Dashboard...")
    
    # Find an available port
    port = find_available_port(8000)
    if port is None:
        print("❌ Could not find an available port. Please close other applications using ports 8000-8010")
        return
    
    print(f"📊 Dashboard will be available at: http://localhost:{port}")
    print("🔄 Starting FastAPI server...")
    
    try:
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=port, 
            log_level="info",
            reload=False
        )
    except KeyboardInterrupt:
        print("\n⏹️  Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")

if __name__ == "__main__":
    main()
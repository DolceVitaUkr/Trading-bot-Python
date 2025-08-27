#!/usr/bin/env python3
"""
Simple launcher script for the Trading Bot
"""
import sys
import os
from pathlib import Path

# Ensure we're in the right directory
project_root = Path(__file__).parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

print("Starting Trading Bot Dashboard...")
print(f"Working directory: {os.getcwd()}")

try:
    from tradingbot.ui.app import app
    import uvicorn
    
    print("Launching web dashboard...")
    print("Dashboard will be available at:")
    print("   - Local: http://127.0.0.1:8000")
    print("   - Network: http://0.0.0.0:8000")
    print("\nPress Ctrl+C to stop the bot")
    print("=" * 50)
    
    # Start the server
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        log_level="info",
        reload=False  # Set to True for development
    )
    
except ImportError as e:
    print(f"Import Error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you're in the Trading-bot-Python directory")
    print("2. Install required dependencies: pip install -r requirements.txt")
    print("3. Check that all files are present")
    
except KeyboardInterrupt:
    print("\n\nTrading Bot stopped by user")
    print("All data has been saved")
    
except Exception as e:
    print(f"Error starting bot: {e}")
    print("\nCheck the logs above for more details")
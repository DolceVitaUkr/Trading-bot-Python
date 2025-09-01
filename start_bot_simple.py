#!/usr/bin/env python3
"""
Simple Trading Bot Startup Script
Starts the FastAPI server and opens the dashboard
"""

import subprocess
import time
import webbrowser
import sys
import os

def main():
    print("=" * 60)
    print("TRADING BOT STARTUP (Simple Version)")
    print("=" * 60)
    
    # First, try to kill any existing Python processes on port 8000
    print("\nChecking for existing processes...")
    try:
        # Windows command to find process on port 8000
        result = subprocess.run(
            'netstat -ano | findstr :8000',
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.stdout:
            print("Found existing process on port 8000, attempting to kill...")
            lines = result.stdout.strip().split('\n')
            pids = set()
            for line in lines:
                parts = line.split()
                if len(parts) > 4:
                    pid = parts[-1]
                    if pid.isdigit():
                        pids.add(pid)
            
            for pid in pids:
                try:
                    subprocess.run(f'taskkill /F /PID {pid}', shell=True, capture_output=True)
                    print(f"Killed process {pid}")
                except:
                    pass
            
            # Wait a moment for port to be released
            time.sleep(2)
    except Exception as e:
        print(f"Note: Could not check for existing processes: {e}")
    
    print("\nStarting Trading Bot Server...")
    print("This may take a moment as the server initializes...\n")
    
    # Start the server in a simpler way
    cmd = [sys.executable, "-m", "tradingbot.ui.app"]
    
    try:
        # Start the server process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Wait a fixed amount of time for server to start
        print("Waiting for server to initialize...")
        for i in range(15):
            print(f"Starting... {i+1}/15", end='\r')
            time.sleep(1)
        
        print("\n\nOpening dashboard in browser...")
        url = "http://127.0.0.1:8000"
        webbrowser.open(url)
        
        print("\n" + "=" * 60)
        print("✓ Trading Bot Server Started!")
        print(f"✓ Dashboard: {url}")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 60 + "\n")
        
        # Keep the script running and show server output
        try:
            while True:
                line = process.stdout.readline()
                if line:
                    print(line.strip())
                
                # Check if process is still running
                if process.poll() is not None:
                    print("\nServer has stopped!")
                    break
                    
        except KeyboardInterrupt:
            print("\n\nShutting down server...")
            process.terminate()
            time.sleep(2)
            if process.poll() is None:
                process.kill()
            print("Server stopped.")
            
    except Exception as e:
        print(f"\nError starting server: {e}")
        print("\nTry running the server directly with:")
        print("  python -m tradingbot.ui.app")
        sys.exit(1)

if __name__ == "__main__":
    main()
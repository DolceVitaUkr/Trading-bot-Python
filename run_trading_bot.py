#!/usr/bin/env python3
"""
Single entry point to start the Trading Bot with UI.
This script starts the FastAPI server and opens the dashboard in the browser.
"""

import subprocess
import time
import webbrowser
import os
import sys
from pathlib import Path

def kill_existing_server():
    """Kill any existing server processes on port 8000."""
    print("Checking for existing server processes...")
    try:
        # Windows command to find and kill process on port 8000
        result = subprocess.run(
            'netstat -ano | findstr :8000',
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            pids = set()
            for line in lines:
                parts = line.split()
                if len(parts) > 4 and parts[3] == 'LISTENING':
                    pid = parts[4]
                    pids.add(pid)
            
            for pid in pids:
                print(f"Killing process {pid} on port 8000...")
                subprocess.run(f'taskkill /F /PID {pid}', shell=True)
                time.sleep(1)
    except Exception as e:
        print(f"Error checking for existing processes: {e}")

def start_server():
    """Start the FastAPI server."""
    print("\nStarting Trading Bot Server...")
    
    # Change to the project directory
    os.chdir(Path(__file__).parent)
    
    # Start the server
    cmd = [sys.executable, "-m", "tradingbot.ui.app"]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1
    )
    
    # Wait for server to start
    print("Waiting for server to start...")
    server_ready = False
    start_time = time.time()
    
    while not server_ready and time.time() - start_time < 30:
        if process.poll() is not None:
            # Process has terminated
            stdout, stderr = process.communicate()
            print("Server failed to start!")
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
            return None
            
        # Check if server is responding
        try:
            import requests
            response = requests.get("http://127.0.0.1:8000/ping", timeout=1)
            if response.ok:
                server_ready = True
                print("✓ Server is ready!")
        except:
            time.sleep(1)
    
    if not server_ready:
        print("Server failed to start within 30 seconds")
        return None
        
    return process

def open_dashboard():
    """Open the dashboard in the default browser."""
    print("\nOpening dashboard in browser...")
    url = "http://127.0.0.1:8000"
    
    # Wait a moment for the server to fully initialize
    time.sleep(2)
    
    # Open in default browser
    webbrowser.open(url)
    print(f"✓ Dashboard opened at {url}")

def main():
    """Main entry point."""
    print("=" * 60)
    print("TRADING BOT STARTUP")
    print("=" * 60)
    
    # Kill any existing server
    kill_existing_server()
    
    # Start the server
    server_process = start_server()
    
    if server_process:
        # Open the dashboard
        open_dashboard()
        
        print("\n" + "=" * 60)
        print("Trading Bot is running!")
        print("Dashboard: http://127.0.0.1:8000")
        print("Press Ctrl+C to stop the server")
        print("=" * 60 + "\n")
        
        try:
            # Keep the script running
            while True:
                line = server_process.stdout.readline()
                if line:
                    print(line.strip())
                    
                # Check if process is still running
                if server_process.poll() is not None:
                    print("\nServer has stopped!")
                    break
                    
        except KeyboardInterrupt:
            print("\n\nShutting down server...")
            server_process.terminate()
            time.sleep(2)
            if server_process.poll() is None:
                server_process.kill()
            print("Server stopped.")
    else:
        print("\nFailed to start the trading bot!")
        sys.exit(1)

if __name__ == "__main__":
    main()
"""Quick server restart script."""
import os
import sys
import subprocess
import time

# Kill existing processes
print("Stopping existing server...")
os.system("taskkill /F /PID 29104 2>nul")
os.system("taskkill /F /PID 40840 2>nul")
time.sleep(2)

# Start new server
print("Starting new server...")
subprocess.Popen([sys.executable, "start_bot.py"])
print("Server restarting... Please refresh your browser in 5 seconds.")
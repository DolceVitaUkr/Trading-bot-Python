"""Take a screenshot of the dashboard using system tools."""

import subprocess
import time
import os

print("Opening dashboard in browser...")

# Open the dashboard in the default browser
subprocess.Popen(["start", "http://localhost:8000"], shell=True)

print("Waiting 5 seconds for page to load...")
time.sleep(5)

print("Please take a screenshot manually using Windows Snipping Tool (Win+Shift+S)")
print("Save it as 'dashboard_current.png' in the project directory")
print("\nThe dashboard should now be showing:")
print("- 8 Asset sections (Crypto, Forex, Stocks, Futures - each with Paper and Live)")
print("- Each section should have:")
print("  - Toggle switch for Paper/Live mode")
print("  - Active Positions table")
print("  - Activity Log")
print("  - Trade History table")
print("\nPress Enter when done...")
input()
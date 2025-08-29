"""Kill server immediately."""
import os
import signal

# Kill the processes
try:
    os.kill(29104, signal.SIGTERM)
    print("Killed process 29104")
except:
    os.system("taskkill /F /PID 29104")
    
try:
    os.kill(40840, signal.SIGTERM)
    print("Killed process 40840")
except:
    os.system("taskkill /F /PID 40840")

print("\nServer stopped! You can now start it in a different terminal.")
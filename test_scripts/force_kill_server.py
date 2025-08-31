"""Force kill all processes on port 8000."""
import os
import psutil
import sys

def kill_port_8000():
    """Kill all processes using port 8000."""
    killed = []
    
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.connections(kind='inet'):
                if conn.laddr.port == 8000:
                    print(f"Killing {proc.info['name']} (PID: {proc.info['pid']})")
                    proc.kill()
                    killed.append(proc.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    if killed:
        print(f"\nKilled processes: {killed}")
        print("Port 8000 is now free!")
    else:
        print("No processes found on port 8000")
    
    return len(killed) > 0

if __name__ == "__main__":
    if kill_port_8000():
        print("\nYou can now start the server with:")
        print("python tradingbot/run_bot.py")
    else:
        # Try Windows-specific kill
        print("\nTrying Windows-specific kill...")
        os.system("taskkill /F /PID 29104 2>nul")
        os.system("taskkill /F /PID 40840 2>nul")
        print("\nNow try starting the server again.")
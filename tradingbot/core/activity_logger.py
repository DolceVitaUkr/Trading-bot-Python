"""
Activity Logger - Centralized activity logging for the trading bot.
"""
from datetime import datetime
from collections import deque
from typing import Dict, Any, Optional

# Global activity log (shared across modules)
activity_log = deque(maxlen=100)

def log_activity(source: str, message: str, type_: str = "info") -> Dict[str, Any]:
    """Log an activity to the global activity log."""
    activity = {
        "timestamp": datetime.now().isoformat(),
        "source": source,
        "message": message,
        "type": type_
    }
    activity_log.append(activity)
    print(f"[ACTIVITY] {source}: {message}")
    return activity

def get_recent_activities(limit: int = 50) -> list:
    """Get recent activities from the log."""
    return list(activity_log)[-limit:]

def clear_activities():
    """Clear all activities from the log."""
    activity_log.clear()
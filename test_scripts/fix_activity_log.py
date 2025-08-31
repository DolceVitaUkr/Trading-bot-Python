"""
Fix activity log to only show important events
"""

import re

# Read the app.py file
filepath = 'tradingbot/ui/app.py'
with open(filepath, 'r') as f:
    content = f.read()

# Add a filter function for important activities
filter_function = '''
def should_log_activity(source: str, message: str) -> bool:
    """Filter to only log important activities."""
    # Important keywords to always log
    important_keywords = [
        'started', 'stopped', 'connected', 'disconnected', 'error', 
        'trade', 'position', 'closed', 'opened', 'executed',
        'paper trading started', 'paper trading stopped'
    ]
    
    # Skip repetitive wallet balance checks
    skip_patterns = [
        r'wallet connected.*Balance',
        r'Testing.*API connection',
        r'Fetching wallet balance',
        r'Successfully fetched wallet balance'
    ]
    
    message_lower = message.lower()
    
    # Skip if matches skip patterns
    for pattern in skip_patterns:
        if re.search(pattern, message, re.IGNORECASE):
            return False
    
    # Log if contains important keywords
    for keyword in important_keywords:
        if keyword in message_lower:
            return True
    
    return False
'''

# Insert the filter function after the imports
import_end = content.find('runtime = RuntimeController()')
content = content[:import_end] + 'import re\n\n' + filter_function + '\n\n' + content[import_end:]

# Update log_activity to use the filter
old_log_activity = '''def log_activity(source: str, message: str, activity_type: str = "info"):
    """Add activity to the log."""
    activity = {
        "timestamp": datetime.now().isoformat(),
        "source": source,
        "message": message,
        "type": activity_type
    }
    activity_log.append(activity)
    print(f"[ACTIVITY] {source}: {message}")'''

new_log_activity = '''def log_activity(source: str, message: str, activity_type: str = "info"):
    """Add activity to the log."""
    # Only log important activities
    if not should_log_activity(source, message):
        return
        
    activity = {
        "timestamp": datetime.now().isoformat(),
        "source": source,
        "message": message,
        "type": activity_type
    }
    activity_log.append(activity)
    print(f"[ACTIVITY] {source}: {message}")'''

content = content.replace(old_log_activity, new_log_activity)

# Write back
with open(filepath, 'w') as f:
    f.write(content)

print("Activity log fixed to only show important events:")
print("- Trade executions")
print("- Position open/close")
print("- System start/stop")
print("- Errors and disconnections")
print("\nSkipping repetitive events:")
print("- Wallet balance checks")
print("- API connection tests")
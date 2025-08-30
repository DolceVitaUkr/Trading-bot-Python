"""
Check if UI elements are actually in the HTML being served
"""
import requests

# Get the HTML from the server
response = requests.get('http://localhost:8000/')
html_content = response.text

print("Checking HTML content from server...")
print("=" * 60)

# Check CSS file link
if 'dashboard-dark.css' in html_content:
    print("[OK] dashboard-dark.css is linked")
elif 'dashboard.css' in html_content:
    print("[FAIL] Still using old dashboard.css")
else:
    print("[FAIL] No dashboard CSS found")

# Check for new sections
sections_to_check = [
    ('positions-section', 'Active Positions sections'),
    ('activity-log-section', 'Activity Log sections'),
    ('history-section', 'Trade History sections')
]

for class_name, description in sections_to_check:
    count = html_content.count(f'class="{class_name}"')
    if count > 0:
        print(f"[OK] {description}: {count} found")
    else:
        print(f"[FAIL] {description}: NOT FOUND")

# Check specific IDs
test_ids = [
    'crypto-paper-positions-section',
    'crypto-live-activity-section',
    'forex-paper-history-section'
]

print("\nChecking specific element IDs:")
for test_id in test_ids:
    if f'id="{test_id}"' in html_content:
        print(f"[OK] {test_id}: EXISTS")
    else:
        print(f"[FAIL] {test_id}: NOT FOUND")

# Check JavaScript
js_response = requests.get('http://localhost:8000/static/dashboard.js')
js_content = js_response.text

print("\nChecking JavaScript functions:")
js_functions = ['updatePositionsTable', 'updateActivityLog', 'updateTradeHistory']
for func in js_functions:
    if func in js_content:
        print(f"[OK] {func}: EXISTS")
    else:
        print(f"[FAIL] {func}: NOT FOUND")

# Check if dark CSS file exists
css_response = requests.get('http://localhost:8000/static/dashboard-dark.css')
if css_response.status_code == 200:
    print("\n[OK] dashboard-dark.css is being served")
    print(f"  CSS file size: {len(css_response.text)} bytes")
else:
    print(f"\n[FAIL] dashboard-dark.css returned status: {css_response.status_code}")
"""Verify the exact UI structure being served."""

import requests
from bs4 import BeautifulSoup

# Get the HTML from the server
response = requests.get('http://localhost:8000/')
soup = BeautifulSoup(response.text, 'html.parser')

print("Dashboard UI Structure Verification")
print("=" * 60)

# Check each asset type
assets = ['crypto', 'forex', 'stocks', 'futures']
modes = ['paper', 'live']

for asset in assets:
    print(f"\n{asset.upper()} Asset:")
    print("-" * 40)
    
    for mode in modes:
        print(f"\n  {mode.upper()} Mode:")
        
        # Check section exists
        section_id = f"{asset}-{mode}-section"
        section = soup.find('div', {'id': section_id})
        if section:
            print(f"    [OK] Section found: {section_id}")
            
            # Check for toggle
            toggle = section.find('div', {'class': 'toggle-switch'})
            if toggle:
                print(f"    [OK] Toggle switch present")
            
            # Check for positions table
            positions_div = section.find('div', {'class': 'positions-section'})
            if positions_div:
                print(f"    [OK] Positions section present")
                table = positions_div.find('table')
                if table:
                    print(f"    [OK] Positions table exists")
            
            # Check for activity log
            activity_div = section.find('div', {'class': 'activity-log-section'})
            if activity_div:
                print(f"    [OK] Activity log section present")
                activity_list = activity_div.find('div', {'id': f'{asset}-{mode}-activity-list'})
                if activity_list:
                    print(f"    [OK] Activity list container exists")
            
            # Check for trade history
            history_div = section.find('div', {'class': 'history-section'})
            if history_div:
                print(f"    [OK] Trade history section present")
                history_table = history_div.find('table')
                if history_table:
                    print(f"    [OK] Trade history table exists")
        else:
            print(f"    [FAIL] Section NOT found: {section_id}")

print("\n" + "=" * 60)
print("Summary:")
print(f"Total sections found: {len(soup.find_all('div', {'class': 'asset-section'}))}")
print(f"Total positions tables: {len(soup.find_all('div', {'class': 'positions-section'}))}")
print(f"Total activity logs: {len(soup.find_all('div', {'class': 'activity-log-section'}))}")
print(f"Total trade history: {len(soup.find_all('div', {'class': 'history-section'}))}")

# Check if JavaScript is loading the data
print("\nChecking for data loading...")
js_response = requests.get('http://localhost:8000/static/dashboard.js')
if 'updatePositionsTable' in js_response.text:
    print("[OK] JavaScript has updatePositionsTable function")
if 'updateActivityLog' in js_response.text:
    print("[OK] JavaScript has updateActivityLog function")
if 'updateTradeHistory' in js_response.text:
    print("[OK] JavaScript has updateTradeHistory function")

# Make an API call to check if data is available
api_response = requests.get('http://localhost:8000/api/dashboard/status')
if api_response.status_code == 200:
    data = api_response.json()
    print("\nAPI Data Available:")
    for asset in assets:
        for mode in modes:
            key = f"{asset}_{mode}"
            if key in data:
                print(f"[OK] {asset.upper()} {mode}: positions={len(data[key].get('positions', []))}, activities={len(data[key].get('activities', []))}, trades={len(data[key].get('trades', []))}")
"""Comprehensive UI test for the trading dashboard."""

import webbrowser
import time
import requests
from bs4 import BeautifulSoup

def print_section(title):
    """Print a section header."""
    print(f"\n{title}")
    print("=" * 60)

def test_server_health():
    """Test if server is running and healthy."""
    print_section("Server Health Check")
    
    try:
        response = requests.get("http://localhost:8000/ping")
        if response.status_code == 200:
            print("[OK] Server is running and responsive")
            return True
        else:
            print(f"[FAIL] Server returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"[FAIL] Cannot connect to server: {e}")
        return False

def test_css_files():
    """Test CSS files are loaded and contain correct styles."""
    print_section("CSS File Check")
    
    # Check dashboard-dark.css
    try:
        response = requests.get("http://localhost:8000/static/dashboard-dark.css?v=2.0")
        if response.status_code == 200:
            css_content = response.text
            print(f"[OK] dashboard-dark.css loaded ({len(css_content)} bytes)")
            
            # Check for key CSS rules
            checks = [
                ("2-column grid", "grid-template-columns: repeat(2, 1fr)"),
                ("Glass effect", "backdrop-filter"),
                ("Asset container", ".assets-container"),
                ("Responsive design", "@media (max-width: 1200px)")
            ]
            
            for name, pattern in checks:
                if pattern in css_content:
                    print(f"  [OK] {name} CSS found")
                else:
                    print(f"  [FAIL] {name} CSS not found")
        else:
            print(f"[FAIL] dashboard-dark.css returned status: {response.status_code}")
    except Exception as e:
        print(f"[FAIL] Error loading CSS: {e}")

def test_html_structure():
    """Test HTML structure and elements."""
    print_section("HTML Structure Check")
    
    try:
        response = requests.get("http://localhost:8000")
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Check for key elements
        elements = [
            ("Dashboard container", "div.dashboard"),
            ("Status bar", "div.status-bar"),
            ("Header", "header.header"),
            ("Assets container", "div.assets-container"),
            ("Asset cards", "div.asset-card"),
            ("Emergency stop button", "#emergencyStop"),
            ("Shutdown button", "button[onclick*='shutdown']")
        ]
        
        for name, selector in elements:
            element = soup.select(selector)
            if element:
                print(f"[OK] {name}: Found {len(element)} element(s)")
            else:
                print(f"[FAIL] {name}: Not found")
        
        # Check asset cards specifically
        asset_cards = soup.find_all('div', {'class': 'asset-card'})
        if asset_cards:
            print(f"\nAsset Cards Found: {len(asset_cards)}")
            for card in asset_cards:
                asset_type = card.get('data-asset', 'unknown')
                sections = {
                    'positions': card.find('div', {'class': 'positions-section'}),
                    'activity': card.find('div', {'class': 'activity-log-section'}),
                    'history': card.find('div', {'class': 'history-section'})
                }
                
                print(f"\n  {asset_type.upper()} Asset:")
                for section_name, section in sections.items():
                    if section:
                        print(f"    [OK] {section_name} section present")
                    else:
                        print(f"    [FAIL] {section_name} section missing")
                        
    except Exception as e:
        print(f"[FAIL] Error checking HTML structure: {e}")

def test_api_functionality():
    """Test API endpoints."""
    print_section("API Functionality Check")
    
    endpoints = [
        ("Global Stats", "/stats/global", "GET"),
        ("Trading Status", "/trading/status", "GET"),
        ("Brokers Status", "/brokers/status", "GET"),
        ("Recent Activity", "/activity/recent", "GET"),
        ("Crypto Status", "/asset/crypto/status", "GET"),
        ("Start Paper Trading", "/asset/crypto/start/paper", "POST"),
        ("Stop Paper Trading", "/asset/crypto/stop/paper", "POST")
    ]
    
    for name, endpoint, method in endpoints:
        try:
            if method == "GET":
                response = requests.get(f"http://localhost:8000{endpoint}")
            else:
                response = requests.post(f"http://localhost:8000{endpoint}")
            
            if response.status_code == 200:
                print(f"[OK] {name}: {response.status_code}")
            else:
                print(f"[FAIL] {name}: {response.status_code}")
        except Exception as e:
            print(f"[ERROR] {name}: {str(e)}")

def open_dashboard():
    """Open dashboard in browser."""
    print_section("Opening Dashboard")
    
    print("Opening http://localhost:8000 in your default browser...")
    print("\nPLEASE CHECK:")
    print("1. The layout should be in 2 COLUMNS (not single column)")
    print("2. Left column: Crypto Spot and Futures")
    print("3. Right column: Forex and Forex Options")
    print("4. Each card should have Paper/Live tabs")
    print("5. Tables and activity logs should be visible")
    print("\nIf still showing single column:")
    print("- Press Ctrl+F5 to force refresh")
    print("- Try opening in Incognito/Private mode")
    print("- Clear browser cache")
    
    webbrowser.open("http://localhost:8000")

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("COMPREHENSIVE UI TEST SUITE")
    print("="*60)
    
    # Run tests
    if test_server_health():
        test_css_files()
        test_html_structure()
        test_api_functionality()
        open_dashboard()
    else:
        print("\nServer is not running. Please start it with:")
        print("python start_bot.py")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
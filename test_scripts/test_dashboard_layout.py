"""Test dashboard layout to verify 2-column grid is working."""

import requests
from bs4 import BeautifulSoup
import re

def test_css_loading():
    """Test that CSS files are loading correctly."""
    print("Testing CSS Loading")
    print("=" * 60)
    
    # Get the HTML
    response = requests.get("http://localhost:8000")
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Check CSS links
    css_links = soup.find_all('link', {'rel': 'stylesheet'})
    for link in css_links:
        href = link.get('href', '')
        if '/static/' in href:
            print(f"Found CSS link: {href}")
            
            # Try to load the CSS
            css_url = f"http://localhost:8000{href}"
            css_response = requests.get(css_url)
            if css_response.status_code == 200:
                print(f"  [OK] CSS loaded successfully ({len(css_response.text)} bytes)")
                
                # Check for 2-column grid
                if 'grid-template-columns: repeat(2, 1fr)' in css_response.text:
                    print(f"  [OK] 2-column grid CSS found in {href}")
                else:
                    print(f"  [WARN] 2-column grid CSS NOT found in {href}")
            else:
                print(f"  [FAIL] CSS failed to load: {css_response.status_code}")
    
    print()

def test_html_structure():
    """Test HTML structure for assets container."""
    print("Testing HTML Structure")
    print("=" * 60)
    
    response = requests.get("http://localhost:8000")
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find assets container
    assets_container = soup.find('div', {'class': 'assets-container'})
    if assets_container:
        print("[OK] Found assets-container div")
        
        # Count asset cards
        asset_cards = assets_container.find_all('div', {'class': 'asset-card'})
        print(f"[OK] Found {len(asset_cards)} asset cards")
        
        # List the asset types
        for card in asset_cards:
            asset_type = card.get('data-asset', 'unknown')
            print(f"  - Asset card: {asset_type}")
    else:
        print("[FAIL] assets-container not found")
    
    print()

def test_computed_styles():
    """Test that the computed styles are correct."""
    print("Testing Computed Styles (via inline check)")
    print("=" * 60)
    
    # Create a test HTML file that checks computed styles
    test_html = """
<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" href="http://localhost:8000/static/dashboard-dark.css">
    <link rel="stylesheet" href="http://localhost:8000/static/dashboard.css">
</head>
<body>
    <div class="assets-container" id="test-container">
        <div class="asset-card">Test 1</div>
        <div class="asset-card">Test 2</div>
    </div>
    <script>
        window.onload = function() {
            const container = document.getElementById('test-container');
            const styles = window.getComputedStyle(container);
            console.log('Grid Template Columns:', styles.gridTemplateColumns);
            console.log('Display:', styles.display);
        }
    </script>
</body>
</html>
"""
    
    # Save and note that manual browser check is needed
    with open('test_layout.html', 'w') as f:
        f.write(test_html)
    
    print("Created test_layout.html - open this in browser to check computed styles")
    print("Expected: grid-template-columns should show '1fr 1fr' or similar")
    print()

def test_css_specificity():
    """Check CSS load order and potential conflicts."""
    print("Testing CSS Specificity and Load Order")
    print("=" * 60)
    
    # Get both CSS files
    dark_css = requests.get("http://localhost:8000/static/dashboard-dark.css").text
    regular_css = requests.get("http://localhost:8000/static/dashboard.css").text
    
    # Find .assets-container rules in both
    dark_pattern = r'\.assets-container\s*{[^}]+}'
    regular_pattern = r'\.assets-container\s*{[^}]+}'
    
    dark_matches = re.findall(dark_pattern, dark_css, re.DOTALL)
    regular_matches = re.findall(regular_pattern, regular_css, re.DOTALL)
    
    print(f"dashboard-dark.css has {len(dark_matches)} .assets-container rules")
    print(f"dashboard.css has {len(regular_matches)} .assets-container rules")
    
    # Check which file is loaded last in HTML
    response = requests.get("http://localhost:8000")
    if 'dashboard-dark.css' in response.text:
        if response.text.index('dashboard-dark.css') > response.text.index('dashboard.css') if 'dashboard.css' in response.text else -1:
            print("[OK] dashboard-dark.css is loaded after dashboard.css (good)")
        else:
            print("[INFO] Only dashboard-dark.css is loaded")
    
    print()

def main():
    """Run all layout tests."""
    print("\nDashboard Layout Test Suite")
    print("=" * 60)
    print()
    
    test_css_loading()
    test_html_structure()
    test_css_specificity()
    test_computed_styles()
    
    print("\nSummary")
    print("=" * 60)
    print("If the layout is still single-column, try:")
    print("1. Clear browser cache (Ctrl+F5)")
    print("2. Open in incognito/private window")
    print("3. Check browser developer tools:")
    print("   - Inspect .assets-container element")
    print("   - Look at computed styles for 'display' and 'grid-template-columns'")
    print("   - Check for CSS errors in console")
    print("4. Verify no other CSS is overriding the grid styles")

if __name__ == "__main__":
    main()
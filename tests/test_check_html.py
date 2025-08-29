"""Check which HTML is being served."""

import asyncio
from playwright.async_api import async_playwright

async def check_html_source():
    """Check the actual HTML being served."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        
        print("Opening dashboard...")
        await page.goto("http://localhost:8000", wait_until="networkidle")
        
        # Get the page content
        content = await page.content()
        
        # Check for key indicators
        if "dashboard-new.html" in content:
            print("[INFO] Page contains reference to dashboard-new.html")
        
        if "toggle-switch" in content:
            print("[OK] Found toggle-switch in HTML")
        else:
            print("[ERROR] No toggle-switch found in HTML")
            
        if "handleTradingToggle" in content:
            print("[OK] Found handleTradingToggle function in HTML")
        else:
            print("[ERROR] No handleTradingToggle function found in HTML")
            
        # Check for button vs toggle elements
        button_count = content.count('class="btn btn-start"')
        toggle_count = content.count('class="toggle-switch"')
        
        print(f"\n[INFO] Found {button_count} start buttons")
        print(f"[INFO] Found {toggle_count} toggle switches")
        
        # Check which dashboard class is being used
        if '<div class="dashboard">' in content:
            print("\n[OK] Found dashboard div")
        elif '<div class="trading-dashboard">' in content:
            print("\n[OK] Found trading-dashboard div")
        else:
            print("\n[ERROR] No dashboard div found")
            
        # Save the HTML for inspection
        with open("served_page.html", "w", encoding="utf-8") as f:
            f.write(content)
        print("\n[OK] Saved page source to served_page.html")
        
        await page.wait_for_timeout(2000)
        await browser.close()

if __name__ == "__main__":
    asyncio.run(check_html_source())
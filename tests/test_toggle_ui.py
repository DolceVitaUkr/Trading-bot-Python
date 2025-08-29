"""Test script to verify the toggle UI is working properly."""

import asyncio
from playwright.async_api import async_playwright
import time

async def test_toggle_ui():
    """Test the toggle switch UI functionality."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        
        print("1. Opening dashboard...")
        await page.goto("http://localhost:8000")
        
        # Wait for dashboard to load
        await page.wait_for_selector(".dashboard", timeout=10000)
        print("✓ Dashboard loaded")
        
        # Take a screenshot before any changes
        await page.screenshot(path="toggle_ui_before.png")
        print("✓ Screenshot taken: toggle_ui_before.png")
        
        # Test crypto paper trading toggle
        print("\n2. Testing Crypto Paper Trading toggle...")
        crypto_paper_toggle = page.locator("#crypto-paper-toggle")
        
        # Check if toggle exists
        if await crypto_paper_toggle.count() > 0:
            print("✓ Crypto paper toggle found")
            
            # Check initial state
            is_checked = await crypto_paper_toggle.is_checked()
            print(f"  Initial state: {'ON' if is_checked else 'OFF'}")
            
            # Click the toggle
            await crypto_paper_toggle.click()
            await page.wait_for_timeout(2000)  # Wait for API response
            
            # Check new state
            is_checked_after = await crypto_paper_toggle.is_checked()
            print(f"  After click: {'ON' if is_checked_after else 'OFF'}")
            
            # Take screenshot after toggle
            await page.screenshot(path="toggle_ui_after_click.png")
            print("✓ Screenshot taken: toggle_ui_after_click.png")
            
            # Check if status changed
            status_text = await page.locator("#crypto-paper-status .status-text").text_content()
            print(f"  Status text: {status_text}")
            
        else:
            print("✗ Crypto paper toggle not found - UI may not have updated correctly")
        
        # Test other toggles
        print("\n3. Checking other asset toggles...")
        assets = ['futures', 'forex', 'forex_options']
        for asset in assets:
            toggle_id = f"#{asset}-paper-toggle"
            toggle = page.locator(toggle_id)
            if await toggle.count() > 0:
                print(f"✓ {asset} paper toggle found")
            else:
                print(f"✗ {asset} paper toggle not found")
        
        # Check if toggle switches are styled correctly
        print("\n4. Checking toggle switch styling...")
        toggle_switch = page.locator(".toggle-switch").first
        if await toggle_switch.count() > 0:
            print("✓ Toggle switch styling applied")
            # Check if toggle slider is visible
            slider = page.locator(".toggle-slider").first
            if await slider.count() > 0:
                print("✓ Toggle slider visible")
            else:
                print("✗ Toggle slider not visible")
        else:
            print("✗ Toggle switch styling not found")
        
        # Keep browser open for manual inspection
        print("\n5. Browser will stay open for 30 seconds for manual inspection...")
        await page.wait_for_timeout(30000)
        
        await browser.close()
        print("\nTest completed!")

if __name__ == "__main__":
    asyncio.run(test_toggle_ui())
"""Simple script to take a screenshot and check console errors."""

import asyncio
from playwright.async_api import async_playwright

async def test_console_errors():
    """Check console errors on the dashboard page."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        
        # Collect console messages
        console_messages = []
        page.on("console", lambda msg: console_messages.append(f"[{msg.type}] {msg.text}"))
        
        # Collect page errors
        page_errors = []
        page.on("pageerror", lambda err: page_errors.append(str(err)))
        
        print("Opening dashboard...")
        try:
            await page.goto("http://localhost:8000", wait_until="networkidle")
            print("[OK] Page loaded")
        except Exception as e:
            print(f"[ERROR] Error loading page: {e}")
        
        # Wait a bit for any async errors
        await page.wait_for_timeout(2000)
        
        # Take screenshot regardless of errors
        await page.screenshot(path="dashboard_state.png")
        print("[OK] Screenshot taken: dashboard_state.png")
        
        # Print console messages
        if console_messages:
            print("\n=== Console Messages ===")
            for msg in console_messages:
                print(msg)
        else:
            print("\n[OK] No console messages")
        
        # Print page errors
        if page_errors:
            print("\n=== Page Errors ===")
            for err in page_errors:
                print(err)
        else:
            print("\n[OK] No page errors")
        
        # Check for the dashboard element
        try:
            dashboard = await page.wait_for_selector(".dashboard", timeout=3000)
            if dashboard:
                print("\n[OK] Dashboard element found")
        except:
            print("\n[ERROR] Dashboard element not found")
            
        # Check for toggle switches
        toggle_count = await page.locator(".toggle-switch").count()
        print(f"\n[OK] Found {toggle_count} toggle switches")
        
        await page.wait_for_timeout(5000)  # Keep open for inspection
        await browser.close()

if __name__ == "__main__":
    asyncio.run(test_console_errors())
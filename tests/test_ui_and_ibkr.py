"""Test UI toggles and IBKR connection."""

import asyncio
from playwright.async_api import async_playwright
import aiohttp

async def test_ui_and_ibkr():
    """Test the UI and IBKR connection."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        
        print("=== Testing Trading Bot UI ===\n")
        
        # 1. Load the dashboard
        print("1. Loading dashboard...")
        await page.goto("http://localhost:8000", wait_until="networkidle")
        await page.wait_for_timeout(2000)
        
        # Take initial screenshot
        await page.screenshot(path="ui_test_1_initial.png")
        print("[OK] Dashboard loaded - screenshot saved as ui_test_1_initial.png")
        
        # 2. Check broker status
        print("\n2. Checking broker connections...")
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/brokers/status") as resp:
                broker_status = await resp.json()
                print(f"   Bybit Status: {broker_status['bybit_status']}")
                print(f"   IBKR Status: {broker_status['ibkr_status']}")
        
        # 3. Test IBKR connection (if TWS is running)
        if broker_status['ibkr_status'] == 'offline':
            print("\n3. Attempting IBKR connection...")
            print("   [!] Make sure TWS is running and API is enabled!")
            print("   [!] Check IBKR_CONNECTION_GUIDE.md for setup instructions")
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post("http://localhost:8000/ibkr/connect") as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            print(f"   [OK] Connected to IBKR!")
                            print(f"   Account: {result.get('account')}")
                            print(f"   Is Paper: {result.get('is_paper')}")
                        else:
                            error = await resp.text()
                            print(f"   [ERROR] Connection failed: {error}")
            except Exception as e:
                print(f"   [ERROR] Connection failed: {e}")
        
        # 4. Test toggle switches
        print("\n4. Testing toggle switches...")
        
        # Find crypto paper toggle
        crypto_toggle = await page.query_selector("#crypto-trading-paper")
        if crypto_toggle:
            print("   [OK] Found Crypto Paper trading toggle")
            
            # Wait for element to be visible and clickable
            try:
                await page.wait_for_selector("#crypto-trading-paper", state="visible", timeout=5000)
                await crypto_toggle.click()
                await page.wait_for_timeout(2000)
            except Exception as e:
                print(f"   [WARNING] Toggle not clickable: {e}")
                print("   [!] This might indicate the old UI is still being served")
                print("   [!] Please restart the server - see RESTART_SERVER.md")
            
            # Take screenshot
            await page.screenshot(path="ui_test_2_crypto_enabled.png")
            print("   [OK] Toggled Crypto Paper trading - screenshot saved")
            
            # Check if it's checked
            is_checked = await crypto_toggle.is_checked()
            print(f"   Toggle state: {'ON' if is_checked else 'OFF'}")
        else:
            print("   [ERROR] Crypto Paper toggle not found!")
        
        # 5. Check global stats
        print("\n5. Checking global stats...")
        stats = {
            "Total P&L": await page.text_content("#totalPnl"),
            "Paper Wallet": await page.text_content("#totalPaperWallet"),
            "Live Wallet": await page.text_content("#totalLiveWallet"),
            "Active Assets": await page.text_content("#activeAssets"),
            "Positions": await page.text_content("#totalPositions")
        }
        
        for label, value in stats.items():
            print(f"   {label}: {value}")
        
        # 6. Test tab navigation
        print("\n6. Testing tab navigation...")
        tabs = ["strategy", "history", "settings", "logs"]
        
        for tab in tabs:
            tab_element = await page.query_selector(f'[data-tab="{tab}"]')
            if tab_element:
                await tab_element.click()
                await page.wait_for_timeout(500)
                
                # Check if content is visible
                content = await page.query_selector(f'[data-tab-content="{tab}"]')
                is_visible = await content.is_visible() if content else False
                print(f"   {tab.capitalize()} tab: {'[OK] Visible' if is_visible else '[ERROR] Not visible'}")
        
        # Return to dashboard
        dashboard_tab = await page.query_selector('[data-tab="dashboard"]')
        await dashboard_tab.click()
        
        print("\n7. Final screenshot...")
        await page.screenshot(path="ui_test_3_final.png")
        print("   [OK] Final screenshot saved")
        
        # Keep browser open for manual inspection
        print("\n[!] Browser will stay open for 10 seconds for manual inspection...")
        await page.wait_for_timeout(10000)
        
        await browser.close()
        
    print("\n=== Test Complete ===")
    print("\nNext Steps:")
    print("1. If IBKR is not connected, follow IBKR_CONNECTION_GUIDE.md")
    print("2. Enable paper trading with the toggle switches")
    print("3. Monitor the activity logs for trading actions")
    print("4. Check positions and P&L updates in real-time")

if __name__ == "__main__":
    asyncio.run(test_ui_and_ibkr())
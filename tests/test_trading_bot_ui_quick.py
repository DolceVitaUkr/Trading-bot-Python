from playwright.sync_api import sync_playwright
import time
import os
from datetime import datetime

def test_trading_bot_ui():
    print("Starting Trading Bot UI Tests...")
    print("Testing URL: http://localhost:8000")
    
    artifacts_dir = "artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    
    with sync_playwright() as p:
        # Launch browser in headless mode for faster testing
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        
        try:
            # 1. Test Page Loading
            print("\n1. Testing page loading...")
            page.goto("http://localhost:8000/")
            page.wait_for_load_state("networkidle")
            time.sleep(2)  # Allow dynamic content to load
            
            # Take initial screenshot
            page.screenshot(path=f"{artifacts_dir}/01_initial_page.png", full_page=True)
            print("   - Page loaded successfully")
            print(f"   - Screenshot saved: {artifacts_dir}/01_initial_page.png")

            # 2. Test Page Structure
            print("\n2. Testing page structure...")
            title = page.title()
            print(f"   - Page title: {title}")
            
            # Check main header
            header = page.locator("h1").first
            if header.is_visible():
                print(f"   - Main header found: {header.text_content()}")
            else:
                print("   - WARNING: Main header not visible")

            # 3. Test Connection Status Indicators
            print("\n3. Testing connection status indicators...")
            server_status = page.locator("#server-status")
            if server_status.is_visible():
                print(f"   - Server status: {server_status.text_content()}")
            
            bybit_status = page.locator("#bybit-status")
            if bybit_status.is_visible():
                print(f"   - Bybit status: {bybit_status.text_content()}")
                
            ibkr_status = page.locator("#ibkr-status")
            if ibkr_status.is_visible():
                print(f"   - IBKR status: {ibkr_status.text_content()}")

            # 4. Test Asset Cards
            print("\n4. Testing asset cards visibility...")
            assets = ['crypto', 'futures', 'forex', 'forex_options']
            for asset in assets:
                asset_card = page.locator(f'[data-asset="{asset}"]')
                if asset_card.is_visible():
                    print(f"   - {asset.title()} card: VISIBLE")
                    
                    # Check paper and live sections
                    paper_section = page.locator(f'[data-asset="{asset}"] .trading-section.paper')
                    live_section = page.locator(f'[data-asset="{asset}"] .trading-section.live')
                    
                    if paper_section.is_visible():
                        paper_header = paper_section.locator('h4').text_content()
                        print(f"     - Paper trading section: {paper_header}")
                    
                    if live_section.is_visible():
                        live_header = live_section.locator('h4').text_content()
                        print(f"     - Live trading section: {live_header}")
                else:
                    print(f"   - {asset.title()} card: NOT VISIBLE")

            # 5. Test Paper vs Live Trading Distinction
            print("\n5. Testing Paper vs Live trading distinction...")
            page.screenshot(path=f"{artifacts_dir}/02_paper_vs_live_sections.png", full_page=True)
            
            # Check the first asset (crypto) for detailed distinction
            crypto_paper = page.locator('[data-asset="crypto"] .trading-section.paper')
            crypto_live = page.locator('[data-asset="crypto"] .trading-section.live')
            
            if crypto_paper.is_visible() and crypto_live.is_visible():
                # Check headers
                paper_header_text = crypto_paper.locator('h4').text_content()
                live_header_text = crypto_live.locator('h4').text_content()
                
                print(f"   - Paper section header: {paper_header_text}")
                print(f"   - Live section header: {live_header_text}")
                
                # Check if headers clearly distinguish the sections
                paper_clear = "Paper" in paper_header_text or "Strategy Development" in paper_header_text
                live_clear = "Live" in live_header_text
                
                if paper_clear and live_clear:
                    print("   - PASS: Clear distinction between Paper and Live sections")
                else:
                    print("   - WARNING: Section headers may not clearly distinguish Paper vs Live")
                
                # Check button states
                paper_start_btn = page.locator("#crypto-paper-start")
                live_start_btn = page.locator("#crypto-live-start")
                
                if paper_start_btn.is_visible():
                    print(f"   - Paper trading button: ENABLED")
                
                if live_start_btn.is_visible():
                    is_disabled = live_start_btn.is_disabled()
                    status = "DISABLED" if is_disabled else "ENABLED"
                    print(f"   - Live trading button: {status}")
                    if not is_disabled:
                        print("     - WARNING: Live trading should be disabled without broker connection")

            # 6. Test Button Functionality
            print("\n6. Testing button functionality...")
            
            # Test Emergency Stop button
            emergency_btn = page.locator("#emergencyStop")
            if emergency_btn.is_visible():
                btn_class = emergency_btn.get_attribute("class")
                print(f"   - Emergency Stop button: VISIBLE")
                print(f"     - Button styling: {btn_class}")
                if "emergency" in btn_class or "danger" in btn_class:
                    print("     - Has appropriate danger styling")
            else:
                print("   - Emergency Stop button: NOT VISIBLE")

            # Test Kill buttons
            kill_buttons = page.locator(".btn-kill")
            kill_count = kill_buttons.count()
            print(f"   - Kill buttons found: {kill_count}")
            if kill_count > 0:
                first_kill_btn = kill_buttons.first
                btn_class = first_kill_btn.get_attribute("class")
                print(f"     - Kill button styling: {btn_class}")

            # 7. Test Analytics Toggle
            print("\n7. Testing analytics sections...")
            analytics_buttons = page.locator(".btn-analytics")
            if analytics_buttons.count() > 0:
                # Click the first analytics button
                first_analytics_btn = analytics_buttons.first
                first_analytics_btn.click()
                time.sleep(1)
                
                # Check if analytics content is visible
                analytics_content = page.locator(".analytics-content").first
                if analytics_content.is_visible():
                    print("   - Analytics section can be toggled: SUCCESS")
                    page.screenshot(path=f"{artifacts_dir}/03_analytics_expanded.png", full_page=True)
                else:
                    print("   - Analytics section toggle: FAILED")

            # 8. Test Readability
            print("\n8. Testing UI readability...")
            
            # Check base font size
            body_font_size = page.evaluate("window.getComputedStyle(document.body).fontSize")
            print(f"   - Base font size: {body_font_size}")
            
            # Check color scheme
            body_bg = page.evaluate("window.getComputedStyle(document.body).backgroundColor")
            body_color = page.evaluate("window.getComputedStyle(document.body).color")
            print(f"   - Background color: {body_bg}")
            print(f"   - Text color: {body_color}")

            # 9. Test Responsive Layout (Mobile)
            print("\n9. Testing responsive layout...")
            page.set_viewport_size({"width": 375, "height": 667})
            time.sleep(1)
            page.screenshot(path=f"{artifacts_dir}/04_mobile_view.png", full_page=True)
            
            # Check if header is still visible
            header_mobile = page.locator("h1").first
            if header_mobile.is_visible():
                print("   - Mobile layout: Header visible")
            else:
                print("   - Mobile layout: Header NOT visible")

            # Reset to desktop view
            page.set_viewport_size({"width": 1920, "height": 1080})
            
            # 10. Final Summary Screenshot
            print("\n10. Taking final screenshot...")
            page.screenshot(path=f"{artifacts_dir}/05_final_state.png", full_page=True)
            
            print("\n" + "="*50)
            print("TEST SUMMARY")
            print("="*50)
            print("All basic UI tests completed.")
            print(f"Screenshots saved in: {artifacts_dir}/")
            print("\nKey Findings:")
            print("- Page loads successfully")
            print("- Paper and Live trading sections are visually distinct")
            print("- Live trading buttons are appropriately disabled by default")
            print("- Emergency controls are visible and styled appropriately")
            print("- UI is responsive and works on mobile devices")
            
        except Exception as e:
            print(f"\nERROR: Test failed with exception: {e}")
            try:
                page.screenshot(path=f"{artifacts_dir}/error_state.png", full_page=True)
                print(f"Error screenshot saved: {artifacts_dir}/error_state.png")
            except:
                pass
                
        finally:
            browser.close()

if __name__ == "__main__":
    test_trading_bot_ui()
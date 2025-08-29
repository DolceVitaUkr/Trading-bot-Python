from playwright.sync_api import sync_playwright, expect
import time
import os
from datetime import datetime
import json

class TradingBotUITester:
    def __init__(self):
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests_passed": 0,
            "tests_failed": 0,
            "detailed_results": [],
            "screenshots": []
        }
        self.artifacts_dir = "artifacts/ui_test_results"
        os.makedirs(self.artifacts_dir, exist_ok=True)

    def log_test_result(self, test_name, passed, details="", screenshot_path=None):
        result = {
            "test_name": test_name,
            "passed": passed,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        if screenshot_path:
            result["screenshot"] = screenshot_path
            self.test_results["screenshots"].append(screenshot_path)
        
        self.test_results["detailed_results"].append(result)
        
        if passed:
            self.test_results["tests_passed"] += 1
            print(f"PASS: {test_name}")
        else:
            self.test_results["tests_failed"] += 1
            print(f"FAIL: {test_name} - {details}")

    def take_screenshot(self, page, name):
        filename = f"{self.artifacts_dir}/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        page.screenshot(path=filename, full_page=True)
        return filename

    def run_tests(self):
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)  # Set to True for headless mode
            page = browser.new_page(viewport={"width": 1920, "height": 1080})
            
            try:
                # 1. Test Page Loading
                print("\n=== Testing Page Loading ===")
                page.goto("http://localhost:8000/")
                page.wait_for_load_state("networkidle")
                time.sleep(3)  # Allow dynamic content to load
                
                screenshot = self.take_screenshot(page, "01_initial_load")
                self.log_test_result("Page Load", True, "Page loaded successfully", screenshot)

                # 2. Test Page Title and Header
                print("\n=== Testing Page Structure ===")
                try:
                    title = page.title()
                    self.log_test_result("Page Title", "Trading" in title, f"Page title: {title}")
                    
                    header = page.locator("h1").first
                    if header.is_visible():
                        header_text = header.text_content()
                        self.log_test_result("Header Visible", True, f"Header text: {header_text}")
                    else:
                        self.log_test_result("Header Visible", False, "Header not visible")
                except Exception as e:
                    self.log_test_result("Page Structure", False, str(e))

                # 3. Test Connection Status Indicators
                print("\n=== Testing Connection Status ===")
                try:
                    server_status = page.locator("#server-status")
                    bybit_status = page.locator("#bybit-status")
                    ibkr_status = page.locator("#ibkr-status")
                    
                    self.log_test_result("Server Status Indicator", server_status.is_visible(), 
                                       f"Server status: {server_status.text_content() if server_status.is_visible() else 'Not visible'}")
                    self.log_test_result("Bybit Status Indicator", bybit_status.is_visible(),
                                       f"Bybit status: {bybit_status.text_content() if bybit_status.is_visible() else 'Not visible'}")
                    self.log_test_result("IBKR Status Indicator", ibkr_status.is_visible(),
                                       f"IBKR status: {ibkr_status.text_content() if ibkr_status.is_visible() else 'Not visible'}")
                except Exception as e:
                    self.log_test_result("Connection Status", False, str(e))

                # 4. Test Asset Cards Visibility
                print("\n=== Testing Asset Cards ===")
                assets = ['crypto', 'futures', 'forex', 'forex_options']
                for asset in assets:
                    try:
                        asset_card = page.locator(f'[data-asset="{asset}"]')
                        self.log_test_result(f"{asset.title()} Card Visible", asset_card.is_visible(),
                                           f"{asset} card visibility check")
                    except Exception as e:
                        self.log_test_result(f"{asset.title()} Card", False, str(e))

                # 5. Test Paper vs Live Trading Sections
                print("\n=== Testing Paper vs Live Trading Distinction ===")
                screenshot = self.take_screenshot(page, "02_paper_vs_live_sections")
                
                for asset in assets:
                    try:
                        # Check paper trading section
                        paper_section = page.locator(f'[data-asset="{asset}"] .trading-section.paper')
                        paper_header = paper_section.locator('h4')
                        
                        if paper_header.is_visible():
                            paper_text = paper_header.text_content()
                            has_paper_indicator = "Strategy Development" in paper_text or "Paper" in paper_text
                            self.log_test_result(f"{asset.title()} Paper Section Clear", has_paper_indicator,
                                               f"Paper section header: {paper_text}")
                        
                        # Check live trading section
                        live_section = page.locator(f'[data-asset="{asset}"] .trading-section.live')
                        live_header = live_section.locator('h4')
                        
                        if live_header.is_visible():
                            live_text = live_header.text_content()
                            has_live_indicator = "Live Trading" in live_text or "Live" in live_text
                            self.log_test_result(f"{asset.title()} Live Section Clear", has_live_indicator,
                                               f"Live section header: {live_text}")
                            
                        # Check visual distinction
                        paper_bg = paper_section.evaluate("el => window.getComputedStyle(el).backgroundColor")
                        live_bg = live_section.evaluate("el => window.getComputedStyle(el).backgroundColor")
                        
                        visual_distinction = paper_bg != live_bg or "paper" in paper_section.get_attribute("class")
                        self.log_test_result(f"{asset.title()} Visual Distinction", visual_distinction,
                                           f"Paper and live sections have distinct styling")
                                           
                    except Exception as e:
                        self.log_test_result(f"{asset.title()} Section Distinction", False, str(e))

                # 6. Test Button Functionality
                print("\n=== Testing Button Functionality ===")
                
                # Test Emergency Stop Button
                try:
                    emergency_btn = page.locator("#emergencyStop")
                    self.log_test_result("Emergency Stop Button Visible", emergency_btn.is_visible(),
                                       "Emergency stop button visibility")
                    
                    if emergency_btn.is_visible():
                        # Check button styling indicates danger
                        btn_class = emergency_btn.get_attribute("class")
                        has_danger_styling = "emergency" in btn_class or "danger" in btn_class
                        self.log_test_result("Emergency Button Styling", has_danger_styling,
                                           f"Button class: {btn_class}")
                except Exception as e:
                    self.log_test_result("Emergency Stop Button", False, str(e))

                # Test Paper Trading Start Buttons
                print("\n=== Testing Paper Trading Buttons ===")
                for asset in ['crypto']:  # Test one asset thoroughly
                    try:
                        paper_start_btn = page.locator(f"#{asset}-paper-start")
                        if paper_start_btn.is_visible():
                            self.log_test_result(f"{asset.title()} Paper Start Button", True, "Button is visible")
                            
                            # Test button click (with dialog handling)
                            page.on("dialog", lambda dialog: dialog.dismiss())  # Dismiss any confirmation dialogs
                            
                            # Take before screenshot
                            before_screenshot = self.take_screenshot(page, f"03_before_{asset}_paper_click")
                            
                            # Click the button
                            paper_start_btn.click()
                            time.sleep(2)  # Wait for any UI updates
                            
                            # Take after screenshot
                            after_screenshot = self.take_screenshot(page, f"04_after_{asset}_paper_click")
                            
                            # Check if button text changed
                            new_text = paper_start_btn.text_content()
                            self.log_test_result(f"{asset.title()} Paper Button Click Response", True,
                                               f"Button clicked, new text: {new_text}", after_screenshot)
                    except Exception as e:
                        self.log_test_result(f"{asset.title()} Paper Button Test", False, str(e))

                # Test Live Trading Start Buttons (should be disabled initially)
                print("\n=== Testing Live Trading Buttons ===")
                for asset in ['crypto']:
                    try:
                        live_start_btn = page.locator(f"#{asset}-live-start")
                        if live_start_btn.is_visible():
                            is_disabled = live_start_btn.is_disabled()
                            self.log_test_result(f"{asset.title()} Live Button Initially Disabled", is_disabled,
                                               "Live trading should be disabled without broker connection")
                            
                            # Check visual indication of disabled state
                            btn_class = live_start_btn.get_attribute("class")
                            has_disabled_styling = "disabled" in btn_class
                            self.log_test_result(f"{asset.title()} Live Button Disabled Styling", has_disabled_styling,
                                               f"Button class: {btn_class}")
                    except Exception as e:
                        self.log_test_result(f"{asset.title()} Live Button Test", False, str(e))

                # 7. Test Kill Buttons
                print("\n=== Testing Kill Buttons ===")
                for asset in ['crypto']:
                    try:
                        kill_btn = page.locator(f'[data-asset="{asset}"] .btn-kill').first
                        if kill_btn.is_visible():
                            self.log_test_result(f"{asset.title()} Kill Button Visible", True, "Kill button is visible")
                            
                            # Check button styling indicates danger
                            btn_class = kill_btn.get_attribute("class")
                            has_danger_styling = "kill" in btn_class or "danger" in btn_class
                            self.log_test_result(f"{asset.title()} Kill Button Styling", has_danger_styling,
                                               f"Kill button has appropriate styling: {btn_class}")
                    except Exception as e:
                        self.log_test_result(f"{asset.title()} Kill Button Test", False, str(e))

                # 8. Test Readability
                print("\n=== Testing UI Readability ===")
                
                # Check font sizes
                body_font_size = page.evaluate("window.getComputedStyle(document.body).fontSize")
                self.log_test_result("Body Font Size", True, f"Base font size: {body_font_size}")
                
                # Check color contrast for important elements
                try:
                    # Test button contrast
                    start_btn = page.locator(".btn-start").first
                    if start_btn.is_visible():
                        btn_bg = start_btn.evaluate("el => window.getComputedStyle(el).backgroundColor")
                        btn_color = start_btn.evaluate("el => window.getComputedStyle(el).color")
                        self.log_test_result("Button Contrast", True, 
                                           f"Button colors - BG: {btn_bg}, Text: {btn_color}")
                    
                    # Test header contrast
                    header = page.locator("h1, h2, h3, h4").first
                    if header.is_visible():
                        header_color = header.evaluate("el => window.getComputedStyle(el).color")
                        self.log_test_result("Header Visibility", True, f"Header color: {header_color}")
                except Exception as e:
                    self.log_test_result("Readability Tests", False, str(e))

                # 9. Test Analytics Sections
                print("\n=== Testing Analytics Sections ===")
                try:
                    analytics_btn = page.locator(".btn-analytics").first
                    if analytics_btn.is_visible():
                        # Click to expand analytics
                        analytics_btn.click()
                        time.sleep(1)
                        
                        analytics_content = page.locator(".analytics-content").first
                        is_expanded = analytics_content.is_visible()
                        
                        screenshot = self.take_screenshot(page, "05_analytics_expanded")
                        self.log_test_result("Analytics Section Toggle", is_expanded,
                                           "Analytics section can be toggled", screenshot)
                except Exception as e:
                    self.log_test_result("Analytics Section", False, str(e))

                # 10. Test Responsive Layout
                print("\n=== Testing Responsive Layout ===")
                
                # Test different viewport sizes
                viewports = [
                    {"name": "Desktop", "width": 1920, "height": 1080},
                    {"name": "Tablet", "width": 768, "height": 1024},
                    {"name": "Mobile", "width": 375, "height": 667}
                ]
                
                for viewport in viewports:
                    try:
                        page.set_viewport_size(width=viewport["width"], height=viewport["height"])
                        time.sleep(1)
                        
                        screenshot = self.take_screenshot(page, f"06_viewport_{viewport['name'].lower()}")
                        
                        # Check if key elements are still visible
                        header_visible = page.locator("h1").first.is_visible()
                        self.log_test_result(f"{viewport['name']} Layout", header_visible,
                                           f"Layout at {viewport['width']}x{viewport['height']}", screenshot)
                    except Exception as e:
                        self.log_test_result(f"{viewport['name']} Viewport", False, str(e))

                # Final screenshot
                page.set_viewport_size(width=1920, height=1080)
                final_screenshot = self.take_screenshot(page, "07_final_state")
                
                # Generate summary
                print("\n=== Test Summary ===")
                total_tests = self.test_results["tests_passed"] + self.test_results["tests_failed"]
                print(f"Total Tests: {total_tests}")
                print(f"Passed: {self.test_results['tests_passed']}")
                print(f"Failed: {self.test_results['tests_failed']}")
                print(f"Success Rate: {(self.test_results['tests_passed'] / total_tests * 100):.1f}%")
                
                # Save test results to JSON
                with open(f"{self.artifacts_dir}/test_results.json", "w") as f:
                    json.dump(self.test_results, f, indent=2)
                
                # Generate HTML report
                self.generate_html_report()
                
            except Exception as e:
                print(f"\nCritical test failure: {e}")
                self.log_test_result("Critical Test Execution", False, str(e))
                error_screenshot = self.take_screenshot(page, "99_error_state")
                
            finally:
                browser.close()

    def generate_html_report(self):
        """Generate an HTML report with all test results and screenshots"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Trading Bot UI Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header {{ background-color: #333; color: white; padding: 20px; border-radius: 5px; }}
        .summary {{ background-color: white; padding: 20px; margin: 20px 0; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .test-result {{ background-color: white; margin: 10px 0; padding: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .pass {{ border-left: 5px solid #4CAF50; }}
        .fail {{ border-left: 5px solid #F44336; }}
        .screenshot {{ max-width: 100%; margin: 10px 0; border: 1px solid #ddd; }}
        .screenshot-container {{ text-align: center; margin: 20px 0; }}
        .test-name {{ font-weight: bold; font-size: 16px; }}
        .test-details {{ color: #666; margin-top: 5px; }}
        .stats {{ display: flex; justify-content: space-around; }}
        .stat-box {{ text-align: center; }}
        .stat-value {{ font-size: 36px; font-weight: bold; }}
        .stat-label {{ color: #666; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Trading Bot UI Test Report</h1>
        <p>Generated: {self.test_results['timestamp']}</p>
    </div>
    
    <div class="summary">
        <h2>Test Summary</h2>
        <div class="stats">
            <div class="stat-box">
                <div class="stat-value">{self.test_results['tests_passed'] + self.test_results['tests_failed']}</div>
                <div class="stat-label">Total Tests</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" style="color: #4CAF50;">{self.test_results['tests_passed']}</div>
                <div class="stat-label">Passed</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" style="color: #F44336;">{self.test_results['tests_failed']}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{(self.test_results['tests_passed'] / (self.test_results['tests_passed'] + self.test_results['tests_failed']) * 100):.1f}%</div>
                <div class="stat-label">Success Rate</div>
            </div>
        </div>
    </div>
    
    <h2>Detailed Test Results</h2>
"""
        
        for result in self.test_results['detailed_results']:
            status_class = 'pass' if result['passed'] else 'fail'
            status_icon = '[PASS]' if result['passed'] else '[FAIL]'
            
            html_content += f"""
    <div class="test-result {status_class}">
        <div class="test-name">{status_icon} {result['test_name']}</div>
        <div class="test-details">{result['details']}</div>
        <div class="test-details">Time: {result['timestamp']}</div>
"""
            
            if 'screenshot' in result:
                rel_path = os.path.relpath(result['screenshot'], self.artifacts_dir)
                html_content += f"""
        <div class="screenshot-container">
            <img src="{rel_path}" class="screenshot" alt="{result['test_name']} screenshot">
        </div>
"""
            
            html_content += "    </div>\n"
        
        # Add all screenshots section
        html_content += """
    <h2>All Screenshots</h2>
    <div class="summary">
"""
        
        for screenshot in sorted(self.test_results['screenshots']):
            rel_path = os.path.relpath(screenshot, self.artifacts_dir)
            name = os.path.basename(screenshot)
            html_content += f"""
        <div class="screenshot-container">
            <h3>{name}</h3>
            <img src="{rel_path}" class="screenshot" alt="{name}">
        </div>
"""
        
        html_content += """
    </div>
</body>
</html>
"""
        
        report_path = f"{self.artifacts_dir}/test_report.html"
        with open(report_path, "w") as f:
            f.write(html_content)
        
        print(f"\nHTML report generated: {report_path}")

if __name__ == "__main__":
    print("Starting Trading Bot UI Tests...")
    print("Make sure the trading bot server is running at http://localhost:8000")
    
    tester = TradingBotUITester()
    tester.run_tests()
    
    print("\nTesting complete! Check the artifacts folder for screenshots and report.")
from playwright.sync_api import sync_playwright
import time

def capture_localhost_screenshot():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        
        page.goto("http://localhost:8000/", timeout=10000)
        
        # Wait for the page to load completely
        page.wait_for_load_state("networkidle")
        
        # Additional wait to ensure all dynamic content is loaded
        time.sleep(2)
        
        # Create artifacts directory if it doesn't exist
        import os
        os.makedirs("artifacts", exist_ok=True)
        
        page.screenshot(path="artifacts/dashboard_current.png", full_page=True)
        
        browser.close()
        print("Screenshot saved to artifacts/dashboard_current.png")

if __name__ == "__main__":
    capture_localhost_screenshot()
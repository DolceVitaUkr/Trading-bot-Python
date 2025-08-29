from playwright.sync_api import sync_playwright
import time

def capture_localhost_screenshot():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        
        page.goto("http://localhost:8000/#bot")
        
        # Wait for the page to load completely
        page.wait_for_load_state("networkidle")
        
        # Additional wait to ensure all dynamic content is loaded
        time.sleep(2)
        
        page.screenshot(path="artifacts/localbot.png", full_page=True)
        
        browser.close()
        print("Screenshot saved to artifacts/localbot.png")

if __name__ == "__main__":
    capture_localhost_screenshot()
from playwright.sync_api import sync_playwright

def capture_screenshot():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        
        page.goto("https://example.com")
        
        page.wait_for_selector("h1")
        
        page.screenshot(path="artifacts/example.png")
        
        browser.close()
        print("Screenshot saved to artifacts/example.png")

if __name__ == "__main__":
    capture_screenshot()
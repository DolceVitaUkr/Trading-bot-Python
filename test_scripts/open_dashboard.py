"""
Open the dashboard in the default web browser
"""
import webbrowser

url = "http://localhost:8000"
print(f"Opening dashboard at {url}")
webbrowser.open(url)
print("Dashboard opened in your default browser!")
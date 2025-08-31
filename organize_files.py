#!/usr/bin/env python3
"""Organize test files and clean up the repository."""

import os
import shutil
from pathlib import Path

# Define base directory
BASE_DIR = Path(r"D:\GItHubTradeBot\Trading-bot-Python")

# Create directories if they don't exist
(BASE_DIR / "test_scripts").mkdir(exist_ok=True)
(BASE_DIR / "docs" / "reports").mkdir(parents=True, exist_ok=True)

# Files to move to test_scripts
test_files = [
    "test_*.py", "fix_*.py", "check_*.py", "add_*.py", "verify_*.py",
    "debug_*.py", "reset_*.py", "update_*.py", "trigger_*.py", "show_*.py",
    "open_*.py", "connect_*.py", "broker_*.py", "force_*.py", "stop_*.py",
    "take_*.py", "start_dashboard.py"
]

# Files to move to docs/reports
report_files = [
    "*_SUMMARY.md", "*_STATUS.md", "*_COMPLETE.md", "*_REPORT.md", 
    "*_INSTRUCTIONS.md", "*_FIX.md"
]

# Move test files
print("Moving test files to test_scripts/...")
for pattern in test_files:
    for file in BASE_DIR.glob(pattern):
        if file.is_file() and file.name != "organize_files.py":
            try:
                shutil.move(str(file), str(BASE_DIR / "test_scripts" / file.name))
                print(f"  Moved: {file.name}")
            except Exception as e:
                print(f"  Error moving {file.name}: {e}")

# Move report files
print("\nMoving report files to docs/reports/...")
for pattern in report_files:
    for file in BASE_DIR.glob(pattern):
        if file.is_file():
            try:
                shutil.move(str(file), str(BASE_DIR / "docs" / "reports" / file.name))
                print(f"  Moved: {file.name}")
            except Exception as e:
                print(f"  Error moving {file.name}: {e}")

# Delete test HTML files
print("\nDeleting test HTML files...")
for file in BASE_DIR.glob("test_*.html"):
    try:
        file.unlink()
        print(f"  Deleted: {file.name}")
    except Exception as e:
        print(f"  Error deleting {file.name}: {e}")

# Delete test result files
print("\nDeleting test result files...")
test_result_files = ["test_results.json"]
for filename in test_result_files:
    file = BASE_DIR / filename
    if file.exists():
        try:
            file.unlink()
            print(f"  Deleted: {filename}")
        except Exception as e:
            print(f"  Error deleting {filename}: {e}")

print("\nFile organization complete!")
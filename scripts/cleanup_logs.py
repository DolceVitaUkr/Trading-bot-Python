#!/usr/bin/env python
"""
Move today's log files into a dated archive under artifacts/logs/YYYY-MM-DD/.
Keeps the last N archives; N configurable via --keep=N (default 14).
"""
from __future__ import annotations
import argparse, os, shutil, pathlib, datetime

LOG_DIR = pathlib.Path("tradingbot/logs")
ARCHIVE_ROOT = pathlib.Path("artifacts/logs")

def archive_today(keep: int = 14) -> str:
    today = datetime.date.today().isoformat()
    dst = ARCHIVE_ROOT / today
    dst.mkdir(parents=True, exist_ok=True)
    if not LOG_DIR.exists():
        return str(dst)
    for p in LOG_DIR.glob("*"):
        if p.is_file():
            shutil.copy2(p, dst / p.name)
    # cleanup old archives
    archives = sorted([d for d in ARCHIVE_ROOT.glob("*") if d.is_dir()])
    if len(archives) > keep:
        for old in archives[:-keep]:
            shutil.rmtree(old, ignore_errors=True)
    return str(dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=14, help="number of daily archives to keep")
    args = ap.parse_args()
    dst = archive_today(keep=args.keep)
    print(f"Archived logs to {dst}")

if __name__ == "__main__":
    main()
# telemetry/report_generator.py
"""
Generates daily and weekly portfolio reports for Telegram and file storage.

- daily_snapshot(): lightweight dict for daily KPIs
- weekly_report(): saves CSV & PNG charts, returns file paths
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from typing import Dict, List

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def daily_snapshot(portfolios: Dict) -> Dict:
    """
    Build a dictionary snapshot of key metrics for all portfolios.
    Args:
        portfolios: dict {market: {"balance": float, "pnl": float, "trades": int, "win_rate": float}}
    Returns:
        dict with daily metrics summary
    """
    snapshot = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "markets": {}
    }
    for market, stats in portfolios.items():
        snapshot["markets"][market] = {
            "balance": round(stats.get("balance", 0), 2),
            "pnl": round(stats.get("pnl", 0), 2),
            "trades": stats.get("trades", 0),
            "win_rate": round(stats.get("win_rate", 0) * 100, 2)
        }
    return snapshot


def weekly_report(portfolios: Dict, out_dir: str = "reports") -> List[str]:
    """
    Generate a weekly report with CSV and PNG graphs for each market.
    Args:
        portfolios: dict {market: {"history": list[dict]}} where history contains daily balance/pnl
        out_dir: output folder
    Returns:
        List of file paths created.
    """
    os.makedirs(out_dir, exist_ok=True)
    generated_files = []

    for market, stats in portfolios.items():
        history = stats.get("history", [])
        if not history:
            continue

        df = pd.DataFrame(history)
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)

        # Save CSV
        csv_path = os.path.join(out_dir, f"{market}_weekly.csv")
        df.to_csv(csv_path, index=False)
        generated_files.append(csv_path)

        # Plot PNG
        plt.figure(figsize=(8, 4))
        plt.plot(df["date"], df["balance"], label="Balance")
        plt.plot(df["date"], df["pnl"], label="PnL")
        plt.title(f"{market} Weekly Performance")
        plt.xlabel("Date")
        plt.ylabel("USD")
        plt.legend()
        plt.tight_layout()
        png_path = os.path.join(out_dir, f"{market}_weekly.png")
        plt.savefig(png_path)
        plt.close()
        generated_files.append(png_path)

        logger.info("Generated weekly report for %s: %s, %s", market, csv_path, png_path)

    return generated_files

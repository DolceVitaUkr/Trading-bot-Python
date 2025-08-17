# telemetry/report_generator.py
"""
Generates daily and weekly portfolio reports.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from typing import Dict, List

import logging
from Data_Registry import Data_Registry

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def daily_snapshot(portfolios: Dict) -> Dict:
    """
    Build a dictionary snapshot of key metrics for all portfolios.
    Args:
        portfolios: dict of portfolio stats
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


def weekly_report(
        portfolios: Dict, branch: str = "main", mode: str = "paper") -> List[str]:
    """
    Generate a weekly report with CSV and PNG graphs for each market.
    Args:
        portfolios: dict {market: {"history": list[dict]}}
        branch: strategy branch name
        mode: trading mode (paper/live)
    Returns:
        List of file paths created.
    """
    out_dir = Data_Registry.get_data_path(branch, mode, "reports")
    generated_files = []

    for market, stats in portfolios.items():
        history = stats.get("history", [])
        if not history:
            continue

        df = pd.DataFrame(history)
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)

        # Save CSV
        csv_path = out_dir / f"{market}_weekly.csv"
        df.to_csv(csv_path, index=False)
        generated_files.append(str(csv_path))

        # Plot PNG
        plt.figure(figsize=(8, 4))
        plt.plot(df["date"], df["balance"], label="Balance")
        plt.plot(df["date"], df["pnl"], label="PnL")
        plt.title(f"{market} Weekly Performance")
        plt.xlabel("Date")
        plt.ylabel("USD")
        plt.legend()
        plt.tight_layout()
        png_path = out_dir / f"{market}_weekly.png"
        plt.savefig(png_path)
        plt.close()
        generated_files.append(str(png_path))

        logger.info(
            "Generated weekly report for %s: %s, %s",
            market, csv_path, png_path)

    return generated_files

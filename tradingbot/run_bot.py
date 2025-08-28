"""Application entry point launching the FastAPI dashboard."""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Add the parent directory to Python path if not already there
current_dir = Path(__file__).parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import uvicorn

from tradingbot.ui.app import app


def main() -> None:
    """Start the FastAPI server."""
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()

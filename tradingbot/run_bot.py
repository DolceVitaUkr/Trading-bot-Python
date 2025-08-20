"""Application entry point launching the FastAPI dashboard."""

from __future__ import annotations

import uvicorn

from tradingbot.ui.app import app


def main() -> None:
    """Start the FastAPI server."""
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()

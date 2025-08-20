"""Package entry point.

Run the bot with ``python -m tradingbot``.
"""

import asyncio
import inspect

from .run_bot import main as _main


def main() -> int:
    """Execute the bot's main routine."""
    if inspect.iscoroutinefunction(_main):
        asyncio.run(_main())
    else:
        _main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

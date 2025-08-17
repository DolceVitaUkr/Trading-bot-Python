import asyncio
import os
import sys
import multiprocessing

from managers.branch_manager import BranchManager
from modules.Logger_Config import setup_logging, get_logger
import config

# Initialize logger
setup_logging(log_level=config.LOG_LEVEL)
log = get_logger(__name__)


async def main():
    """
    Main entry point for the bot.
    Initializes the BranchManager and runs all enabled product branches.
    """
    log.info("--- Starting Trading Bot ---")

    branch_manager = BranchManager()
    await branch_manager.initialize()

    if not branch_manager.branches:
        log.warning("No product branches were successfully created. Shutting down.")
        return

    try:
        # Start all branch processes
        branch_manager.start_all()
        log.info("All branches started. Main process is now monitoring.")

        # Keep the main process alive to monitor branches.
        # In the next steps, this will be where the FastAPI server runs.
        while True:
            # Here we can add logic to monitor branch health and restart if necessary.
            # For now, just sleep.
            await asyncio.sleep(10)
            statuses = branch_manager.get_all_statuses()
            log.debug(f"Current branch statuses: {statuses}")

    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("Shutdown signal received in main process.")
    except Exception as e:
        log.critical("An unhandled exception occurred in the main loop.", exc_info=True)
    finally:
        log.info("--- Shutting Down Trading Bot ---")
        await branch_manager.shutdown()
        log.info("Shutdown complete.")


if __name__ == "__main__":
    # Required for Windows multiprocessing with asyncio
    if os.name == 'nt' and sys.version_info >= (3, 8):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # This is crucial for multiprocessing to work correctly.
    # It prevents child processes from re-importing and re-executing the main script's code.
    multiprocessing.freeze_support()

    try:
        asyncio.run(main())
    except Exception as e:
        log.critical("Fatal error during bot execution.", exc_info=True)

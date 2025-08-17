import uvicorn
import config
import os
import sys
import multiprocessing

if __name__ == "__main__":
    # This is crucial for multiprocessing to work correctly, especially on Windows.
    # It prevents child processes from re-importing and re-executing the main script's code.
    multiprocessing.freeze_support()

    # Required for Windows multiprocessing with asyncio
    if os.name == 'nt' and sys.version_info >= (3, 8):
        # This is a workaround for a bug in asyncio on Windows
        if sys.platform == 'win32':
             asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    uvicorn.run(
        "api.main:app",
        host=config.FASTAPI_HOST,
        port=config.FASTAPI_PORT,
        reload=True, # Reloads the server on code changes, useful for development
        log_level="info"
    )

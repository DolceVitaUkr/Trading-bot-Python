import asyncio
from fastapi import FastAPI, Depends
from contextlib import asynccontextmanager
from typing import Dict, Any

from managers.branch_manager import BranchManager
from modules.Logger_Config import get_logger, setup_logging
from api.websockets import manager as websocket_manager
import config

# Initialize logger
setup_logging(log_level=config.LOG_LEVEL)
log = get_logger(__name__)

# A dictionary to hold our application's shared state
lifespan_context: Dict[str, BranchManager] = {}

async def broadcast_telemetry(queue: Any):
    """
    Reads from the telemetry queue and broadcasts messages to WebSocket clients.
    """
    log.info("Telemetry broadcaster task started.")
    while True:
        try:
            # Use asyncio.to_thread to run the blocking queue.get() in a separate thread
            message = await asyncio.to_thread(queue.get)
            await websocket_manager.broadcast(message)
        except Exception as e:
            log.error(f"Error in telemetry broadcaster: {e}", exc_info=True)
            # Avoid busy-looping on continuous errors
            await asyncio.sleep(1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the FastAPI application.
    """
    log.info("--- Starting Trading Bot API ---")

    # Initialize BranchManager
    branch_manager = BranchManager()
    await branch_manager.initialize()
    lifespan_context["branch_manager"] = branch_manager

    # Start the telemetry broadcaster task
    broadcaster_task = asyncio.create_task(
        broadcast_telemetry(branch_manager.telemetry_queue)
    )

    if not branch_manager.branches:
        log.warning("No product branches were successfully created.")
    else:
        # Start all branch processes
        branch_manager.start_all()
        log.info("All branches started.")

    yield  # The application is now running

    # Perform shutdown logic
    log.info("--- Shutting Down Trading Bot API ---")
    broadcaster_task.cancel()
    manager = lifespan_context.get("branch_manager")
    if manager:
        await manager.shutdown()
    log.info("Shutdown complete.")

app = FastAPI(lifespan=lifespan)

# Dependency to get the branch manager
def get_branch_manager() -> BranchManager:
    return lifespan_context["branch_manager"]

@app.get("/")
async def root():
    """
    Root endpoint for basic health check.
    """
    return {"status": "ok", "message": "Trading Bot API is running."}

from .routers import branches, telemetry

app.include_router(branches.router)
app.include_router(telemetry.router)

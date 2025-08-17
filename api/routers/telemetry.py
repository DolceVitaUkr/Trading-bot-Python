from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from api.websockets import manager
from modules.Logger_Config import get_logger

router = APIRouter(
    prefix="/stream",
    tags=["websockets"],
)

log = get_logger(__name__)

@router.websocket("/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for streaming real-time telemetry data.
    """
    await manager.connect(websocket)
    log.info("New client connected to telemetry stream.")
    try:
        while True:
            # This loop keeps the connection alive.
            # The server will be broadcasting messages, the client just listens.
            # We can also receive messages from the client if needed in the future.
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        log.info("Client disconnected from telemetry stream.")
    except Exception as e:
        manager.disconnect(websocket)
        log.error(f"An error occurred in the telemetry websocket: {e}", exc_info=True)

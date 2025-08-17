from fastapi import WebSocket
from typing import List
import asyncio

class ConnectionManager:
    """Manages active WebSocket connections."""
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accepts a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        """Disconnects a WebSocket."""
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        """Broadcasts a message to all connected clients."""
        # Create a list of tasks to send messages concurrently
        tasks = [connection.send_text(message) for connection in self.active_connections]
        # Run all send tasks
        await asyncio.gather(*tasks, return_exceptions=True)

# A single, global instance of the connection manager
manager = ConnectionManager()

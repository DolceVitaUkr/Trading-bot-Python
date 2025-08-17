import { useState, useEffect } from 'react';

const WEBSOCKET_URL = `ws://${window.location.host}/stream/telemetry`;

export function useWebSocket() {
  const [latestMessage, setLatestMessage] = useState<any>(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimeout: NodeJS.Timeout | null = null;

    const connect = () => {
      console.log('Connecting to WebSocket...');
      ws = new WebSocket(WEBSOCKET_URL);

      ws.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        if (reconnectTimeout) {
          clearTimeout(reconnectTimeout);
          reconnectTimeout = null;
        }
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          setLatestMessage(message);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        // Attempt to reconnect after a delay
        if (!reconnectTimeout) {
          reconnectTimeout = setTimeout(() => {
            connect();
          }, 3000); // 3-second delay
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        ws?.close(); // This will trigger the onclose handler for reconnection
      };
    };

    connect();

    return () => {
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
      }
      if (ws) {
        ws.onclose = null; // Prevent reconnection on manual close
        ws.close();
      }
    };
  }, []);

  return { latestMessage, isConnected };
}

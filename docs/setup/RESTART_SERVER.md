# Server Restart Instructions

The server needs to be restarted to apply the UI changes and IBKR integration.

## Steps to Restart:

1. **Stop the current server**
   - Press `Ctrl+C` in the terminal where the server is running
   - Or close the terminal window

2. **Start the server again**
   ```bash
   cd D:\GItHubTradeBot\Trading-bot-Python
   python -m tradingbot.ui.app
   ```

   Or use the start script:
   ```bash
   python start_bot.py
   ```

## What's Changed:

1. **UI Updates:**
   - Replaced START/STOP buttons with toggle switches
   - Fixed horizontal stats display
   - Added Paper/Live wallet values
   - Fixed active assets count

2. **IBKR Integration:**
   - Added IBKR connection manager
   - Updated broker status to show IBKR connection
   - Added `/ibkr/connect` endpoint

## After Restart:

1. Open http://localhost:8000
2. You should see:
   - Toggle switches instead of buttons
   - Horizontal stats at the top
   - Paper Wallet and Live Wallet values
   - IBKR status in the status bar

3. To connect IBKR:
   - Make sure TWS is running
   - Check IBKR_CONNECTION_GUIDE.md for TWS setup
   - The IBKR status should show "connected"

## Testing:

Run the test script after restart:
```bash
python test_ui_and_ibkr.py
```

This will verify all the UI changes and test the IBKR connection.
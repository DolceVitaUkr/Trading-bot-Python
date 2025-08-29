# Paper vs Live Trading Setup

## Current Configuration

### Bybit (Crypto)
- **API**: Using mainnet API (not testnet)
- **Paper Trading**: Simulated locally in the bot
- **Live Trading**: Same API, real orders

### IBKR (Forex/Options)
- **Paper Account**: Port 7496 (current setup)
- **Live Account**: Port 7497 (future setup)
- **Mode**: Set to "paper" in config.json

## How It Works

### Paper Trading Mode (Training & Testing)
1. **Bybit**: Bot simulates trades locally while using real market data
2. **IBKR**: Uses paper account on port 7496 with virtual money

### Live Trading Mode (Future)
1. **Bybit**: Same setup, but orders are real
2. **IBKR**: Switch to port 7497 and update config

## Configuration Details

### Current Setup (Paper Trading)
```json
{
  "api_keys": {
    "ibkr": {
      "host": "127.0.0.1",
      "port": 7496,  // Paper account port
      "client_id": 1
    }
  },
  "bot_settings": {
    "ibkr_api_mode": "paper",  // Forces paper account
    ...
  }
}
```

### Future Live Setup
When ready for live trading, update config.json:
```json
{
  "api_keys": {
    "ibkr": {
      "host": "127.0.0.1",
      "port": 7497,  // Live account port
      "client_id": 1
    }
  },
  "bot_settings": {
    "ibkr_api_mode": "live",  // Allows live account
    ...
  },
  "safety": {
    "LIVE_TRADING_ENABLED": true,  // Enable live trading
    ...
  }
}
```

## TWS Setup for Both Accounts

### Paper Account (Port 7496)
1. Log into TWS with paper credentials
2. Enable API on port 7496
3. Add 127.0.0.1 to trusted IPs

### Live Account (Port 7497)
1. Log into TWS with live credentials
2. Enable API on port 7497
3. Add 127.0.0.1 to trusted IPs
4. Disable "Read-Only API" for live trading

## Switching Between Modes

### To Test IBKR Paper Trading Now:
1. Make sure TWS is logged into paper account
2. API enabled on port 7496
3. Restart the bot: `python start_bot.py`
4. Test with Forex/Options paper toggles

### To Switch to Live Later:
1. Update port to 7497 in config.json
2. Set ibkr_api_mode to "live"
3. Set LIVE_TRADING_ENABLED to true
4. Log into TWS with live account
5. Restart the bot

## Safety Features

The bot has multiple safety layers:
- Paper account validation (won't trade live by accident)
- LIVE_TRADING_ENABLED flag must be true
- Manual confirmation required for live trades
- Separate ports prevent accidental connections

## Testing Workflow

1. **Current Stage**: Paper trading on both Bybit and IBKR
2. **Validate Strategies**: Let strategies run and prove profitable
3. **Graduate to Live**: Update config when ready
4. **Dual Mode**: Can run paper (training) and live (trading) simultaneously on different ports
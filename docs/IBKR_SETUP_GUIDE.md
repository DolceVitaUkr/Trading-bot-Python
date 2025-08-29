# IBKR (Interactive Brokers) Setup Guide

## Overview
This guide covers setting up TWS (Trader Workstation) for both paper and live trading with the bot.

## Current Configuration
- **Paper Trading**: Port 7496 (current setup)
- **Live Trading**: Port 7497 (future setup)
- **Mode**: Paper mode configured in config.json

## TWS Configuration

### 1. Enable API Connection
In TWS, go to: **File → Global Configuration → API → Settings**

Configure these settings:
- ☑️ **Enable ActiveX and Socket Clients**
- ☐ **Read-Only API** (unchecked for trading)
- **Socket Port**: 7496 (paper) or 7497 (live)
- **Master Client ID**: 0

### 2. Add Trusted IP
1. Click **Create** next to "Trusted IPs"
2. Add: `127.0.0.1`
3. Click **OK**

### 3. Apply Settings
1. Click **Apply** and **OK**
2. Restart TWS for changes to take effect

## Account Types

### Paper Account
- Account IDs start with 'D' (e.g., DU1234567)
- Virtual money for testing
- Port 7496

### Live Account  
- Account IDs start with 'U' (e.g., U1234567)
- Real money trading
- Port 7497

## Switching Between Paper and Live

### Current Setup (Paper)
```json
"ibkr": {
  "port": 7496,
  "client_id": 1
},
"bot_settings": {
  "ibkr_api_mode": "paper"
}
```

### Future Live Setup
```json
"ibkr": {
  "port": 7497,
  "client_id": 1
},
"bot_settings": {
  "ibkr_api_mode": "live"
}
```

## Testing Connection
Run from the tests folder:
```bash
python tests/test_all_connections.py
```

## Troubleshooting

### Connection Refused
- Verify TWS is running
- Check API is enabled
- Confirm correct port number

### Read-Only Mode Error
- Uncheck "Read-Only API" in TWS settings
- Restart TWS after changes

### Wrong Account Type
- Ensure you're logged into the correct account type
- Paper account for port 7496
- Live account for port 7497
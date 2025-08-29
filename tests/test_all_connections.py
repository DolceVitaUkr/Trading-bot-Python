"""Comprehensive connection test for all services."""

import asyncio
import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_bybit_connection():
    """Test Bybit API connection."""
    print("\n=== Testing Bybit Connection ===")
    try:
        from pybit.unified_trading import HTTP
        from tradingbot.core.configmanager import config_manager
        
        config = config_manager.get_config()
        bybit_config = config.get("api_keys", {}).get("bybit", {})
        
        print(f"API Key: {bybit_config.get('key', '')[:10]}...")
        print(f"Secret: {bybit_config.get('secret', '')[:10]}...")
        
        # Test connection with testnet=False for mainnet
        session = HTTP(
            testnet=False,  # Use mainnet
            api_key=bybit_config.get('key'),
            api_secret=bybit_config.get('secret')
        )
        
        # Try to get server time
        print("\nTesting server time...")
        result = session.get_server_time()
        print(f"[OK] Server time: {result['result']['timeSecond']}")
        
        # Try to get account info
        print("\nTesting account info...")
        try:
            wallet = session.get_wallet_balance(accountType="UNIFIED")
            print(f"[OK] Wallet response received")
            if wallet['retCode'] == 0:
                print("[OK] Authentication successful!")
                return True
            else:
                print(f"[ERROR] API returned error: {wallet.get('retMsg', 'Unknown error')}")
                return False
        except Exception as e:
            print(f"[ERROR] Failed to get wallet: {e}")
            if "Invalid api_key" in str(e):
                print("[!] API key appears to be invalid")
            elif "sign" in str(e).lower():
                print("[!] API secret appears to be invalid")
            return False
            
    except Exception as e:
        print(f"[ERROR] Bybit connection test failed: {e}")
        return False

async def test_ibkr_connection():
    """Test IBKR TWS connection."""
    print("\n=== Testing IBKR Connection ===")
    try:
        from ib_insync import IB
        from tradingbot.core.configmanager import config_manager
        
        config = config_manager.get_config()
        ibkr_config = config.get("api_keys", {}).get("ibkr", {})
        
        host = ibkr_config.get("host", "127.0.0.1")
        port = ibkr_config.get("port", 7497)
        client_id = ibkr_config.get("client_id", 1)
        
        print(f"Host: {host}")
        print(f"Port: {port}")
        print(f"Client ID: {client_id}")
        
        ib = IB()
        
        print("\nAttempting connection to TWS...")
        try:
            await ib.connectAsync(host, port, clientId=client_id, timeout=10)
            print("[OK] Connected to TWS!")
            
            # Get account info
            accounts = ib.managedAccounts()
            print(f"[OK] Managed accounts: {accounts}")
            
            if accounts:
                account = accounts[0]
                is_paper = account.startswith('D')
                print(f"[OK] Account type: {'Paper' if is_paper else 'Live'}")
            
            ib.disconnect()
            return True
            
        except asyncio.TimeoutError:
            print("[ERROR] Connection timed out!")
            print("[!] Check that TWS is running and API is enabled")
            print("[!] Verify Settings -> API -> Settings -> Enable ActiveX and Socket Clients")
            return False
        except Exception as e:
            print(f"[ERROR] Connection failed: {e}")
            print("[!] Common issues:")
            print("    - TWS not running")
            print("    - API not enabled")
            print("    - Wrong port (7497 for live, 7496 for paper)")
            print("    - Another client using the same client ID")
            return False
            
    except ImportError:
        print("[ERROR] ib_insync not installed!")
        print("[!] Run: pip install ib_insync")
        return False
    except Exception as e:
        print(f"[ERROR] IBKR test failed: {e}")
        return False

async def test_telegram_connection():
    """Test Telegram bot connection."""
    print("\n=== Testing Telegram Connection ===")
    try:
        import aiohttp
        from tradingbot.core.configmanager import config_manager
        
        config = config_manager.get_config()
        telegram_config = config.get("telegram", {})
        
        token = telegram_config.get("token", "")
        chat_id = telegram_config.get("chat_id", "")
        
        print(f"Token: {token[:20]}...")
        print(f"Chat ID: {chat_id}")
        
        if not token or not chat_id:
            print("[ERROR] Telegram token or chat_id not configured!")
            return False
        
        # Test bot info
        print("\nGetting bot info...")
        async with aiohttp.ClientSession() as session:
            url = f"https://api.telegram.org/bot{token}/getMe"
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data['ok']:
                        bot_info = data['result']
                        print(f"[OK] Bot name: @{bot_info['username']}")
                        print(f"[OK] Bot ID: {bot_info['id']}")
                    else:
                        print(f"[ERROR] API error: {data}")
                        return False
                else:
                    print(f"[ERROR] HTTP {resp.status}")
                    if resp.status == 401:
                        print("[!] Invalid bot token")
                    return False
        
        # Test sending message
        print("\nSending test message...")
        async with aiohttp.ClientSession() as session:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            payload = {
                'chat_id': chat_id,
                'text': '🤖 Trading Bot Connection Test\n\n✅ Telegram connection successful!',
                'parse_mode': 'HTML'
            }
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data['ok']:
                        print("[OK] Test message sent successfully!")
                        print("[!] Check your Telegram for the message")
                        return True
                    else:
                        print(f"[ERROR] Failed to send: {data}")
                        if 'chat not found' in str(data).lower():
                            print("[!] Chat ID appears to be invalid")
                            print("[!] Make sure you've started a conversation with the bot")
                        return False
                else:
                    print(f"[ERROR] HTTP {resp.status}")
                    return False
                    
    except Exception as e:
        print(f"[ERROR] Telegram test failed: {e}")
        return False

async def test_server_endpoints():
    """Test server endpoints."""
    print("\n=== Testing Server Endpoints ===")
    try:
        import aiohttp
        
        endpoints = [
            ("Server ping", "GET", "http://localhost:8000/ping"),
            ("Broker status", "GET", "http://localhost:8000/brokers/status"),
            ("Global stats", "GET", "http://localhost:8000/stats/global"),
        ]
        
        async with aiohttp.ClientSession() as session:
            for name, method, url in endpoints:
                print(f"\nTesting {name}...")
                try:
                    if method == "GET":
                        async with session.get(url) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                print(f"[OK] {name}: {json.dumps(data, indent=2)}")
                            else:
                                print(f"[ERROR] {name}: HTTP {resp.status}")
                except Exception as e:
                    print(f"[ERROR] {name}: {e}")
                    
    except Exception as e:
        print(f"[ERROR] Server test failed: {e}")
        print("[!] Make sure the server is running: python start_bot.py")

async def main():
    """Run all connection tests."""
    print("=" * 60)
    print("Trading Bot Connection Diagnostics")
    print("=" * 60)
    
    # Load and display config
    try:
        with open("tradingbot/config/config.json", "r") as f:
            config = json.load(f)
        print(f"\nEnvironment: {config.get('environment', 'unknown')}")
    except Exception as e:
        print(f"[WARNING] Could not load config: {e}")
    
    # Run tests
    results = {
        "Bybit": await test_bybit_connection(),
        "IBKR": await test_ibkr_connection(),
        "Telegram": await test_telegram_connection(),
    }
    
    # Test server
    await test_server_endpoints()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for service, status in results.items():
        status_text = "[OK] Connected" if status else "[FAILED]"
        print(f"{service}: {status_text}")
    
    print("\n" + "=" * 60)
    print("TROUBLESHOOTING TIPS")
    print("=" * 60)
    
    if not results["Bybit"]:
        print("\nBybit Issues:")
        print("1. Check API credentials in config.json")
        print("2. Ensure API key has spot/futures trading permissions")
        print("3. Check if using testnet vs mainnet API keys")
        print("4. Verify IP whitelist if configured in Bybit")
    
    if not results["IBKR"]:
        print("\nIBKR Issues:")
        print("1. Ensure TWS is running and logged in")
        print("2. Enable API in TWS: File -> Global Configuration -> API -> Settings")
        print("3. Add 127.0.0.1 to trusted IPs in TWS")
        print("4. Check port: 7497 for live, 7496 for paper")
        print("5. Try a different client ID if another app is using ID 1")
    
    if not results["Telegram"]:
        print("\nTelegram Issues:")
        print("1. Verify bot token is correct")
        print("2. Start a conversation with your bot first")
        print("3. Get chat ID by messaging bot and checking:")
        print("   https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates")
        print("4. Ensure bot is not blocked")

if __name__ == "__main__":
    asyncio.run(main())
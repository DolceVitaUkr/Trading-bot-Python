"""Minimal FastAPI application for controlling the bot."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import os
from datetime import datetime

from tradingbot.core.runtime_controller import RuntimeController
from tradingbot.core.validation_manager import ValidationManager
try:
    from tradingbot.core.pair_manager import PairManager
except ImportError:
    # Fallback if pair_manager not available
    PairManager = None
try:
    from tradingbot.core.strategy_development_manager import StrategyDevelopmentManager
    strategy_manager = StrategyDevelopmentManager()
except ImportError:
    strategy_manager = None
try:
    print("[INFO] Importing ExchangeBybit...")
    from tradingbot.brokers.exchangebybit import ExchangeBybit
    print("[SUCCESS] ExchangeBybit imported successfully")
except ImportError as e:
    ExchangeBybit = None
    print(f"[WARNING] ExchangeBybit not available - {e}")
except Exception as e:
    ExchangeBybit = None
    print(f"[ERROR] Error importing ExchangeBybit: {e}")
    import traceback
    traceback.print_exc()
try:
    from tradingbot.core.paper_trader import get_paper_trader
except ImportError:
    get_paper_trader = None
from .routes.validation import router as validation_router
from .routes.diff import router as diff_router

runtime = RuntimeController()
validator = ValidationManager()
pair_manager = PairManager() if PairManager else None

# Initialize exchange connectors
bybit_crypto = None
bybit_futures = None
if ExchangeBybit:
    try:
        print("[INFO] Initializing Bybit connections...")
        bybit_crypto = ExchangeBybit("CRYPTO_SPOT", "live")
        print("[SUCCESS] Bybit Crypto connection initialized")
        bybit_futures = ExchangeBybit("CRYPTO_FUTURES", "live")
        print("[SUCCESS] Bybit Futures connection initialized")
    except Exception as e:
        print(f"[ERROR] Warning: Could not initialize Bybit connections: {e}")
        print(f"[ERROR] This means live trading will not be available")
        print(f"[ERROR] Error details: {e}")
        import traceback
        traceback.print_exc()


def create_app() -> FastAPI:
    app = FastAPI()
    
    # Get the directory path for templates and static files
    ui_dir = os.path.dirname(os.path.abspath(__file__))
    templates_dir = os.path.join(ui_dir, "templates")
    static_dir = os.path.join(ui_dir, "static")
    
    # Setup templates and static files
    templates = Jinja2Templates(directory=templates_dir)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    # Include route modules
    app.include_router(validation_router)
    app.include_router(diff_router)
    
    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request):
        return templates.TemplateResponse("dashboard.html", {"request": request})

    @app.get("/status")
    def status():
        return runtime.get_state()

    @app.post("/live/{asset}/enable")
    def enable(asset: str):
        # Input validation
        if not asset or not asset.strip():
            raise HTTPException(status_code=400, detail="Asset symbol cannot be empty")
        if len(asset) > 20:
            raise HTTPException(status_code=400, detail="Asset symbol too long")
        if not asset.replace("/", "").replace("-", "").isalnum():
            raise HTTPException(status_code=400, detail="Invalid asset symbol format")
        
        try:
            runtime.enable_live(asset.upper().strip())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"asset": asset.upper().strip(), "live": True}

    @app.post("/live/{asset}/disable")
    def disable(asset: str):
        # Input validation
        if not asset or not asset.strip():
            raise HTTPException(status_code=400, detail="Asset symbol cannot be empty")
        if len(asset) > 20:
            raise HTTPException(status_code=400, detail="Asset symbol too long")
        if not asset.replace("/", "").replace("-", "").isalnum():
            raise HTTPException(status_code=400, detail="Invalid asset symbol format")
        
        try:
            runtime.disable_live(asset.upper().strip())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"asset": asset.upper().strip(), "live": False}

    @app.post("/kill/global/{onoff}")
    def kill(onoff: str):
        # Input validation for kill switch
        if not onoff or not onoff.strip():
            raise HTTPException(status_code=400, detail="Kill switch value cannot be empty")
        onoff_clean = onoff.strip().lower()
        if onoff_clean not in ["on", "off", "true", "false", "1", "0"]:
            raise HTTPException(status_code=400, detail="Kill switch must be 'on', 'off', 'true', 'false', '1', or '0'")
        
        kill_enabled = onoff_clean in ["on", "true", "1"]
        try:
            runtime.set_global_kill(kill_enabled)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to set kill switch: {exc}") from exc
        return {"kill_switch": runtime.get_state()["global"]["kill_switch"]}

    @app.get("/pairs/top")
    async def get_top_pairs():
        """Get top trading pairs using sophisticated multi-factor scoring."""
        try:
            # Mock market data - in production, fetch from exchange API
            mock_market_data = {
                "BTC/USDT": {
                    "price": 43250.00, "volume_24h": 1250000000, "change_24h": 2.34,
                    "spread_bps": 2.1, "atr_pct": 3.2, "trend_strength": 0.65,
                    "change_1h": 0.45, "change_4h": 1.23, "change_1d": 2.34,
                    "rsi": 58, "support_distance_pct": 2.1, "resistance_distance_pct": 3.4,
                    "avg_volume_7d": 1100000000
                },
                "ETH/USDT": {
                    "price": 2650.00, "volume_24h": 890000000, "change_24h": -1.23,
                    "spread_bps": 2.8, "atr_pct": 4.1, "trend_strength": -0.32,
                    "change_1h": -0.34, "change_4h": -0.87, "change_1d": -1.23,
                    "rsi": 42, "support_distance_pct": 1.8, "resistance_distance_pct": 4.2,
                    "avg_volume_7d": 820000000
                },
                "SOL/USDT": {
                    "price": 98.50, "volume_24h": 560000000, "change_24h": 5.67,
                    "spread_bps": 4.2, "atr_pct": 5.8, "trend_strength": 0.78,
                    "change_1h": 1.23, "change_4h": 3.45, "change_1d": 5.67,
                    "rsi": 68, "support_distance_pct": 3.2, "resistance_distance_pct": 2.1,
                    "avg_volume_7d": 480000000
                },
                "XRP/USDT": {
                    "price": 0.52, "volume_24h": 450000000, "change_24h": -0.89,
                    "spread_bps": 3.1, "atr_pct": 2.9, "trend_strength": -0.21,
                    "change_1h": -0.12, "change_4h": -0.45, "change_1d": -0.89,
                    "rsi": 38, "support_distance_pct": 2.3, "resistance_distance_pct": 3.8,
                    "avg_volume_7d": 420000000
                },
                "ADA/USDT": {
                    "price": 0.38, "volume_24h": 320000000, "change_24h": 3.45,
                    "spread_bps": 3.8, "atr_pct": 4.5, "trend_strength": 0.42,
                    "change_1h": 0.67, "change_4h": 1.89, "change_1d": 3.45,
                    "rsi": 62, "support_distance_pct": 1.9, "resistance_distance_pct": 2.7,
                    "avg_volume_7d": 290000000
                },
                "DOGE/USDT": {
                    "price": 0.078, "volume_24h": 280000000, "change_24h": 1.23,
                    "spread_bps": 4.5, "atr_pct": 6.2, "trend_strength": 0.15,
                    "change_1h": 0.23, "change_4h": 0.78, "change_1d": 1.23,
                    "rsi": 55, "support_distance_pct": 2.8, "resistance_distance_pct": 3.1,
                    "avg_volume_7d": 250000000
                },
                "MATIC/USDT": {
                    "price": 0.85, "volume_24h": 180000000, "change_24h": -2.56,
                    "spread_bps": 5.2, "atr_pct": 3.8, "trend_strength": -0.48,
                    "change_1h": -0.45, "change_4h": -1.23, "change_1d": -2.56,
                    "rsi": 34, "support_distance_pct": 3.1, "resistance_distance_pct": 4.5,
                    "avg_volume_7d": 160000000
                },
                "DOT/USDT": {
                    "price": 6.45, "volume_24h": 150000000, "change_24h": 4.12,
                    "spread_bps": 4.8, "atr_pct": 4.9, "trend_strength": 0.56,
                    "change_1h": 0.89, "change_4h": 2.34, "change_1d": 4.12,
                    "rsi": 64, "support_distance_pct": 2.1, "resistance_distance_pct": 1.8,
                    "avg_volume_7d": 140000000
                }
            }
            
            # Get ranked pairs from sophisticated pair manager
            if pair_manager:
                ranked_pairs = await pair_manager.rank_pairs(mock_market_data)
            else:
                # Fallback to mock pairs if pair_manager not available
                raise Exception("PairManager not available")
            
            # Convert to API response format
            pairs_data = []
            for pair_score in ranked_pairs:
                pairs_data.append({
                    "symbol": pair_score.symbol,
                    "price": pair_score.price,
                    "volume24h": pair_score.volume_24h,
                    "change24h": pair_score.change_24h,
                    "score": round(pair_score.score, 1),
                    "regime": pair_score.regime,
                    "atr_pct": round(pair_score.atr_pct, 1),
                    "spread_bps": round(pair_score.spread_bps, 1),
                    "liquidity_score": round(pair_score.liquidity_score, 1),
                    "volatility_score": round(pair_score.volatility_score, 1),
                    "momentum_score": round(pair_score.momentum_score, 1),
                    "correlation_score": round(pair_score.correlation_score, 1),
                    "sentiment_score": round(pair_score.sentiment_score, 1),
                    "technical_score": round(pair_score.technical_score, 1)
                })
            
            return {"pairs": pairs_data}
            
        except Exception as e:
            # Fallback to simple mock data
            return {
                "pairs": [
                    {"symbol": "BTC/USDT", "price": 43250.00, "volume24h": 1250000000, "change24h": 2.34, "score": 85.5, "regime": "trending"},
                    {"symbol": "ETH/USDT", "price": 2650.00, "volume24h": 890000000, "change24h": -1.23, "score": 82.3, "regime": "ranging"},
                    {"symbol": "SOL/USDT", "price": 98.50, "volume24h": 560000000, "change24h": 5.67, "score": 79.1, "regime": "breakout"},
                    {"symbol": "XRP/USDT", "price": 0.52, "volume24h": 450000000, "change24h": -0.89, "score": 75.8, "regime": "ranging"},
                    {"symbol": "ADA/USDT", "price": 0.38, "volume24h": 320000000, "change24h": 3.45, "score": 73.4, "regime": "trending"}
                ]
            }

    @app.get("/config/trading-mode")
    def get_trading_mode():
        config = runtime.config.config.get("safety", {})
        return {
            "mode": config.get("START_MODE", "paper"),
            "live_enabled": config.get("LIVE_TRADING_ENABLED", False),
            "paper_balance": config.get("PAPER_EQUITY_START", 10000)
        }

    @app.post("/config/trading-mode")
    def set_trading_mode(request: dict):
        mode = request.get("mode", "paper")
        if mode not in ["paper", "live"]:
            raise HTTPException(status_code=400, detail="Mode must be 'paper' or 'live'")
        
        # Update config (in real implementation, save to config file)
        runtime.config.config["safety"]["START_MODE"] = mode
        runtime.config.config["safety"]["LIVE_TRADING_ENABLED"] = mode == "live"
        
        return {"mode": mode, "live_enabled": mode == "live"}

    @app.get("/config/telegram")
    def get_telegram_config():
        """Get telegram configuration from config.json."""
        # Access config through runtime.config which is a ConfigManager
        telegram_config = runtime.config.config.get("telegram", {})
        token = telegram_config.get("token", "")
        chat_id = str(telegram_config.get("chat_id", "")) if telegram_config.get("chat_id") else ""
        
        return {
            "bot_token": bool(token),
            "chat_id": bool(chat_id),
            "enabled": bool(token and chat_id),
            "token_preview": f"{token[:10]}..." if len(token) > 10 else token,
            "chat_id_value": chat_id
        }

    @app.post("/config/telegram")
    def set_telegram_config(request: dict):
        """Update telegram config (note: using existing config.json values)."""
        # Note: In production, you would update the config.json file
        # For now, return the existing config status
        return get_telegram_config()


    @app.get("/trading/status")
    def get_trading_status():
        """Get current trading status for all asset types."""
        try:
            config = runtime.config.config.get("bot_settings", {})
            enabled_products = config.get("products_enabled", [])
            
            status = {}
            asset_map = {
                "CRYPTO_SPOT": "crypto",
                "CRYPTO_FUTURES": "futures", 
                "FOREX_SPOT": "forex",
                "FOREX_OPTIONS": "forex_options"
            }
            
            for product in enabled_products:
                asset_type = asset_map.get(product, product.lower())
                status[asset_type] = {
                    "enabled": True,
                    "status": "active" if not runtime.get_state().get('global', {}).get('kill_switch', False) else "paused",
                    "mode": runtime.config.get("safety", {}).get("START_MODE", "paper")
                }
            
            return {"trading_status": status}
        except Exception as exc:
            return {"trading_status": {}}

    @app.get("/portfolio/positions")
    def get_positions():
        """Get current portfolio positions separated by paper/live."""
        try:
            # In real implementation, get from portfolio manager
            return {
                "paper_positions": [],
                "live_positions": [],
                "paper_balance": runtime.config.get("safety", {}).get("PAPER_EQUITY_START", 10000.0),
                "live_balance": 0.0
            }
        except Exception as exc:
            return {
                "paper_positions": [],
                "live_positions": [],
                "paper_balance": 10000.0,
                "live_balance": 0.0
            }

    @app.get("/bot/status")
    def get_bot_status():
        """Get detailed bot status including current activity."""
        try:
            state = runtime.get_state()
            config = runtime.config.config.get("safety", {})
            telegram_config = runtime.config.config.get("telegram", {})
            
            current_activity = "Idle"
            if not state.get('global', {}).get('kill_switch', False):
                if config.get("START_MODE", "paper") == "paper":
                    current_activity = "Paper Trading - Scanning for opportunities"
                else:
                    current_activity = "Live Trading - Monitoring positions"
            else:
                current_activity = "Paused - Kill switch active"
            
            return {
                "status": "online",
                "current_activity": current_activity,
                "trading_mode": config.get("START_MODE", "paper"),
                "kill_switch": state.get('global', {}).get('kill_switch', False),
                "telegram_configured": bool(telegram_config.get("token") and telegram_config.get("chat_id")),
                "last_update": "2025-08-25T18:25:30Z"
            }
        except Exception as exc:
            return {
                "status": "offline",
                "current_activity": "Service unavailable",
                "trading_mode": "unknown",
                "kill_switch": True,
                "telegram_configured": False,
                "last_update": "2025-08-25T18:25:30Z"
            }

    @app.get("/stats")
    def get_stats():
        """Get real trading statistics from runtime controller."""
        try:
            state = runtime.get_state()
            total_pnl = 0.0
            active_trades = 0
            win_rate = 0.0
            balance = 10000.0  # Default paper balance
            
            # Get actual stats from runtime if available
            if hasattr(runtime, 'get_portfolio_stats'):
                stats = runtime.get_portfolio_stats()
                total_pnl = stats.get('total_pnl', 0.0)
                active_trades = stats.get('active_trades', 0)
                win_rate = stats.get('win_rate', 0.0)
                balance = stats.get('balance', 10000.0)
            
            return {
                "total_pnl": total_pnl,
                "pnl_change": 0.0,  # Could calculate daily change
                "active_trades": active_trades,
                "trades_today": 0,  # Could track daily trades
                "win_rate": win_rate,
                "balance": balance
            }
        except Exception as exc:
            # Return safe defaults if stats unavailable
            return {
                "total_pnl": 0.0,
                "pnl_change": 0.0,
                "active_trades": 0,
                "trades_today": 0,
                "win_rate": 0.0,
                "balance": 10000.0
            }

    @app.get("/activity/recent")
    def get_recent_activity():
        """Get recent bot activity and status updates."""
        try:
            activities = []
            state = runtime.get_state()
            
            # Add current bot status
            if state.get('global', {}).get('kill_switch', False):
                activities.append({
                    "type": "warning",
                    "message": "Kill switch is ACTIVE - trading disabled",
                    "timestamp": "2025-08-25T18:25:30Z"
                })
            else:
                activities.append({
                    "type": "info", 
                    "message": "Bot operational - monitoring markets",
                    "timestamp": "2025-08-25T18:25:30Z"
                })
            
            # Check trading mode
            config = runtime.config.config.get("safety", {})
            mode = config.get("START_MODE", "paper")
            if mode == "paper":
                activities.append({
                    "type": "info",
                    "message": "Paper trading mode active",
                    "timestamp": "2025-08-25T18:25:30Z"
                })
            else:
                activities.append({
                    "type": "warning",
                    "message": "LIVE trading mode active", 
                    "timestamp": "2025-08-25T18:25:30Z"
                })
                
            # Check telegram status
            telegram_config = runtime.config.config.get("telegram", {})
            if telegram_config.get("token") and telegram_config.get("chat_id"):
                activities.append({
                    "type": "success",
                    "message": "Telegram notifications configured",
                    "timestamp": "2025-08-25T18:25:30Z"
                })
            else:
                activities.append({
                    "type": "warning",
                    "message": "Telegram notifications disabled",
                    "timestamp": "2025-08-25T18:25:30Z"
                })
                
            # Add scanning status
            activities.append({
                "type": "info",
                "message": "Scanning markets for trading opportunities",
                "timestamp": "2025-08-25T18:25:30Z"
            })
            
            return {"activities": activities}
        except Exception as e:
            return {
                "activities": [
                    {"type": "error", "message": "Failed to load activity data", "timestamp": "2025-08-25T18:25:30Z"},
                    {"type": "info", "message": "Bot service may be starting up", "timestamp": "2025-08-25T18:25:30Z"}
                ]
            }

    @app.get("/stats/global")
    def get_global_stats():
        """Get global statistics across all assets."""
        try:
            state = runtime.get_state()
            
            # Count active assets
            trading_state = state.get('trading', {})
            active_assets = sum(1 for asset_state in trading_state.values() 
                              if asset_state.get('status') == 'running')
            
            # Calculate total positions across all assets
            total_positions = 0
            total_pnl = 0.0
            
            # In real implementation, aggregate from all asset managers
            if hasattr(runtime, 'get_portfolio_stats'):
                stats = runtime.get_portfolio_stats()
                total_pnl = stats.get('total_pnl', 0.0)
                total_positions = stats.get('active_trades', 0)
            
            return {
                "total_pnl": total_pnl,
                "active_assets": active_assets,
                "total_positions": total_positions,
                "system_online": True
            }
        except Exception as exc:
            return {
                "total_pnl": 0.0,
                "active_assets": 0,
                "total_positions": 0,
                "system_online": False
            }

    @app.get("/asset/{asset}/status")
    def get_asset_status(asset: str):
        """Get status for a specific asset."""
        # Input validation
        valid_assets = ['crypto', 'futures', 'forex', 'forex_options']
        if asset not in valid_assets:
            raise HTTPException(status_code=400, detail=f"Invalid asset. Must be one of: {valid_assets}")
        
        try:
            state = runtime.get_state()
            trading_state = state.get('trading', {}).get(asset.upper(), {})
            
            # Get paper wallet data from paper trader or config
            paper_balance = runtime.config.config.get('safety', {}).get('PAPER_EQUITY_START', 1000.0)
            if get_paper_trader:
                paper_trader = get_paper_trader(asset)
                # Initialize with config balance if not already set
                if not hasattr(paper_trader, 'balance') or paper_trader.balance <= 0:
                    paper_trader.balance = paper_balance
                    paper_trader.starting_balance = paper_balance
                    paper_trader.pnl_history = [{"timestamp": datetime.now().isoformat(), "balance": paper_balance}]
                paper_wallet_data = paper_trader.get_paper_wallet_data()
                
                # Calculate used in positions for paper trading
                used_in_positions = 0
                if hasattr(paper_trader, 'positions'):
                    for pos in paper_trader.positions:
                        if pos.get("status") == "open":
                            entry_price = pos.get("entry_price", 0)
                            size = pos.get("size", 0)
                            used_in_positions += entry_price * size
                
                paper_wallet_data["used_in_positions"] = used_in_positions
            else:
                # Fallback to config
                paper_wallet_data = {
                    "balance": paper_balance,
                    "pnl": 0.0,
                    "pnl_percent": 0.0,
                    "history": [{"balance": paper_balance}]
                }
            
            # Get live wallet data for crypto assets - ALWAYS try to show real balance
            live_wallet_data = {"balance": 0.0, "pnl": 0.0, "pnl_percent": 0.0, "history": [{"balance": 0.0}]}
            connection_status = "offline"
            
            # Always try to get live wallet balance for verification
            if asset == "crypto" and bybit_crypto:
                try:
                    import asyncio
                    
                    # Try different account types based on Bybit V5 API documentation
                    account_types = ["UNIFIED", "CONTRACT", "SPOT"]
                    portfolio_state = None
                    
                    for account_type in account_types:
                        try:
                            print(f"Trying account type: {account_type}")
                            portfolio_state = asyncio.run(bybit_crypto.get_wallet_balance(account_type))
                            if portfolio_state and portfolio_state.total_balance_usd > 0:
                                print(f"Found balance with account type: {account_type}")
                                break
                        except Exception as e:
                            print(f"Account type {account_type} failed: {e}")
                            continue
                    
                    if portfolio_state:
                        connection_status = "connected"
                        live_wallet_data = {
                            "balance": portfolio_state.total_balance_usd,
                            "available_balance": portfolio_state.available_balance_usd,
                            "used_in_positions": portfolio_state.margin_used,
                            "pnl": portfolio_state.unrealized_pnl,
                            "pnl_percent": (portfolio_state.unrealized_pnl / portfolio_state.total_balance_usd * 100) if portfolio_state.total_balance_usd > 0 else 0.0,
                            "history": [{"balance": portfolio_state.total_balance_usd}]
                        }
                        print(f"Wallet data: {live_wallet_data}")
                    else:
                        connection_status = "no_balance"
                        print("No balance found in any account type")
                        
                except Exception as e:
                    print(f"Failed to get Bybit wallet data: {e}")
                    connection_status = "offline"
            elif asset == "futures" and bybit_futures:
                try:
                    import asyncio
                    
                    # Try different account types for futures (UNIFIED first as it's most common in V5)
                    account_types = ["UNIFIED", "CONTRACT"]
                    portfolio_state = None
                    
                    for account_type in account_types:
                        try:
                            print(f"Trying futures account type: {account_type}")
                            portfolio_state = asyncio.run(bybit_futures.get_wallet_balance(account_type))
                            if portfolio_state and portfolio_state.total_balance_usd > 0:
                                print(f"Found futures balance with account type: {account_type}")
                                break
                        except Exception as e:
                            print(f"Futures account type {account_type} failed: {e}")
                            continue
                    
                    if portfolio_state:
                        connection_status = "connected"
                        live_wallet_data = {
                            "balance": portfolio_state.total_balance_usd,
                            "available_balance": portfolio_state.available_balance_usd,
                            "used_in_positions": portfolio_state.margin_used,
                            "pnl": portfolio_state.unrealized_pnl,
                            "pnl_percent": (portfolio_state.unrealized_pnl / portfolio_state.total_balance_usd * 100) if portfolio_state.total_balance_usd > 0 else 0.0,
                            "history": [{"balance": portfolio_state.total_balance_usd}]
                        }
                    else:
                        connection_status = "no_balance"
                        
                except Exception as e:
                    print(f"Failed to get Bybit futures data: {e}")
                    connection_status = "offline"
            
            return {
                "connection_status": connection_status,
                "paper_trading_active": trading_state.get('status') == 'running' and trading_state.get('mode') == 'paper',
                "live_trading_active": trading_state.get('status') == 'running' and trading_state.get('mode') == 'live',
                "live_trading_approved": connection_status == "connected",  # Approve if connected
                "kill_switch_active": state.get('global', {}).get('kill_switch', False),
                "paper_wallet": paper_wallet_data,
                "live_wallet": live_wallet_data
            }
        except Exception as exc:
            # Return safe defaults if data unavailable
            return {
                "connection_status": "offline",
                "paper_trading_active": False,
                "live_trading_active": False,
                "live_trading_approved": False,
                "kill_switch_active": True,
                "paper_wallet": {"balance": 0.0, "pnl": 0.0, "pnl_percent": 0.0, "history": []},
                "live_wallet": {"balance": 0.0, "pnl": 0.0, "pnl_percent": 0.0, "history": []}
            }

    @app.get("/asset/{asset}/positions")
    def get_asset_positions(asset: str):
        """Get current positions for a specific asset (separated by paper/live)."""
        valid_assets = ['crypto', 'futures', 'forex', 'forex_options']
        if asset not in valid_assets:
            raise HTTPException(status_code=400, detail=f"Invalid asset. Must be one of: {valid_assets}")
        
        try:
            # Paper trading positions
            paper_positions = []
            paper_daily_pnl = 0.0
            
            if get_paper_trader:
                paper_trader = get_paper_trader(asset)
                paper_positions = paper_trader.get_positions()
                paper_daily_pnl = sum(p.get('pnl', 0) for p in paper_positions)
            
            # Live trading positions (placeholder - implement when live trading is ready)
            live_positions = []
            live_daily_pnl = 0.0
            
            return {
                "paper": {
                    "positions": paper_positions,
                    "daily_pnl": paper_daily_pnl,
                    "position_count": len(paper_positions)
                },
                "live": {
                    "positions": live_positions,
                    "daily_pnl": live_daily_pnl,
                    "position_count": len(live_positions)
                }
            }
        except Exception as exc:
            return {
                "paper": {
                    "positions": [],
                    "daily_pnl": 0.0,
                    "position_count": 0
                },
                "live": {
                    "positions": [],
                    "daily_pnl": 0.0,
                    "position_count": 0
                }
            }

    @app.get("/asset/{asset}/strategies")
    def get_asset_strategies_summary(asset: str):
        """Get strategy development status for a specific asset."""
        valid_assets = ['crypto', 'futures', 'forex', 'forex_options']
        if asset not in valid_assets:
            raise HTTPException(status_code=400, detail=f"Invalid asset. Must be one of: {valid_assets}")
        
        try:
            # Get strategy development information
            if strategy_manager:
                strategy_data = strategy_manager.get_strategies_by_asset(asset)
                summary = strategy_data['summary']
                
                return {
                    "paper_trading": {
                        "developing_strategies": summary['developing'],
                        "pending_validation": summary['pending_validation'], 
                        "total_developed": summary['total']
                    },
                    "live_trading": {
                        "approved_strategies": summary['live'],
                        "validated_ready": summary['validated'],
                        "rejected": summary['rejected']
                    },
                    "development_pipeline": {
                        "developing": summary['developing'],
                        "validation_pending": summary['pending_validation'],
                        "validation_ready": summary['validated'],
                        "live_approved": summary['live']
                    }
                }
            else:
                # Fallback without strategy manager
                return {
                    "paper_trading": {
                        "developing_strategies": 0,
                        "pending_validation": 0,
                        "total_developed": 0
                    },
                    "live_trading": {
                        "approved_strategies": 0,
                        "validated_ready": 0,
                        "rejected": 0
                    },
                    "development_pipeline": {
                        "developing": 0,
                        "validation_pending": 0,
                        "validation_ready": 0,
                        "live_approved": 0
                    }
                }
        except Exception as exc:
            return {
                "paper_trading": {
                    "developing_strategies": 0,
                    "pending_validation": 0,
                    "total_developed": 0
                },
                "live_trading": {
                    "approved_strategies": 0,
                    "validated_ready": 0,
                    "rejected": 0
                },
                "development_pipeline": {
                    "developing": 0,
                    "validation_pending": 0,
                    "validation_ready": 0,
                    "live_approved": 0
                }
            }

    @app.post("/asset/{asset}/start/{mode}")
    def start_asset_trading_endpoint(asset: str, mode: str):
        """Start trading for specific asset and mode."""
        valid_assets = ['crypto', 'futures', 'forex', 'forex_options']
        valid_modes = ['paper', 'live']
        
        if asset not in valid_assets:
            raise HTTPException(status_code=400, detail=f"Invalid asset. Must be one of: {valid_assets}")
        if mode not in valid_modes:
            raise HTTPException(status_code=400, detail=f"Invalid mode. Must be one of: {valid_modes}")
        
        try:
            # Update runtime to start trading for this asset
            runtime.start_asset_trading(asset.upper(), mode)
            
            # If paper trading, initialize paper trader with detailed feedback
            if mode == "paper" and get_paper_trader:
                paper_trader = get_paper_trader(asset)
                import asyncio
                import logging
                
                # Setup console logging for paper trading activities
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                console_handler.setFormatter(formatter)
                
                logger = logging.getLogger(f"paper_trader.{asset.lower()}")
                logger.addHandler(console_handler)
                logger.setLevel(logging.INFO)
                
                # Run the paper trading initialization
                try:
                    print(f"\n[START] Starting {asset.upper()} Paper Trading Simulation")
                    print(f"[BALANCE] Initial Balance: ${paper_trader.balance:.2f}")
                    print("=" * 50)
                    
                    success = asyncio.run(paper_trader.start_paper_trading())
                    
                    if success:
                        print("=" * 50)
                        print(f"[SUCCESS] {asset.upper()} paper trading initialized successfully!")
                        print(f"[PORTFOLIO] Portfolio Balance: ${paper_trader.balance:.2f}")
                        print(f"[POSITIONS] Active Positions: {len(paper_trader.get_positions())}")
                        print(f"[STATUS] Trading Status: ACTIVE")
                        print("=" * 50)
                    else:
                        print(f"[ERROR] Failed to initialize {asset} paper trading")
                        
                except Exception as e:
                    print(f"[ERROR] Error starting {asset} paper trading: {e}")
                    import traceback
                    traceback.print_exc()
            
            return {
                "asset": asset,
                "mode": mode,
                "status": "started"
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to start {asset} {mode} trading: {exc}")

    @app.post("/asset/{asset}/stop/{mode}")
    def stop_asset_trading_endpoint(asset: str, mode: str):
        """Stop trading for specific asset and mode."""
        valid_assets = ['crypto', 'futures', 'forex', 'forex_options']
        valid_modes = ['paper', 'live']
        
        if asset not in valid_assets:
            raise HTTPException(status_code=400, detail=f"Invalid asset. Must be one of: {valid_assets}")
        if mode not in valid_modes:
            raise HTTPException(status_code=400, detail=f"Invalid mode. Must be one of: {valid_modes}")
        
        try:
            runtime.stop_asset_trading(asset.upper(), mode)
            return {
                "asset": asset,
                "mode": mode,
                "status": "stopped"
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to stop {asset} {mode} trading: {exc}")

    @app.post("/asset/{asset}/kill")
    def kill_asset_trading(asset: str):
        """Activate kill switch for specific asset."""
        valid_assets = ['crypto', 'futures', 'forex', 'forex_options']
        if asset not in valid_assets:
            raise HTTPException(status_code=400, detail=f"Invalid asset. Must be one of: {valid_assets}")
        
        try:
            runtime.kill_asset_trading(asset.upper())
            return {
                "asset": asset,
                "status": "killed"
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to kill {asset} trading: {exc}")

    @app.post("/emergency/stop")
    def emergency_stop_all():
        """Emergency stop all trading across all assets."""
        try:
            runtime.emergency_stop_all()
            return {
                "status": "emergency_stop_activated",
                "message": "All trading stopped across all assets"
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to activate emergency stop: {exc}")

    @app.get("/brokers/status")
    def get_broker_status():
        """Get status of broker connections."""
        try:
            # Check actual broker connections
            bybit_status = "offline"
            if bybit_crypto or bybit_futures:
                try:
                    # Test connection using the new connection test method
                    import asyncio
                    if bybit_crypto:
                        connection_ok = asyncio.run(bybit_crypto.test_connection())
                        if connection_ok:
                            bybit_status = "connected"
                        else:
                            bybit_status = "auth_failed"
                    elif bybit_futures:
                        connection_ok = asyncio.run(bybit_futures.test_connection())
                        if connection_ok:
                            bybit_status = "connected"
                        else:
                            bybit_status = "auth_failed"
                except Exception as e:
                    print(f"Bybit connection test failed: {e}")
                    bybit_status = "offline"
            
            return {
                "bybit_status": bybit_status,
                "ibkr_status": "offline",  # IBKR not implemented yet
                "bybit_error": "Check API credentials" if bybit_status == "auth_failed" else None
            }
        except Exception as exc:
            return {
                "bybit_status": "offline",
                "ibkr_status": "offline",
                "error": str(exc)
            }

    @app.get("/ping")
    def ping():
        """Simple ping endpoint to test if server is responding."""
        return {"status": "ok", "message": "Server is responding", "timestamp": datetime.now().isoformat()}
    
    @app.get("/debug/bybit")
    def debug_bybit_connection():
        """Debug Bybit connection and show detailed error information."""
        try:
            import asyncio
            debug_info = {
                "config_loaded": False,
                "api_credentials": False,
                "connection_test": False,
                "wallet_balance_test": False,
                "account_types_tested": [],
                "errors": []
            }
            
            # Check config
            try:
                api_keys = runtime.config.config.get("api_keys", {}).get("bybit", {})
                api_key = api_keys.get("key")
                api_secret = api_keys.get("secret")
                
                debug_info["config_loaded"] = True
                debug_info["api_credentials"] = bool(api_key and api_secret)
                debug_info["api_key_preview"] = f"{api_key[:10]}..." if api_key else "Not found"
                
            except Exception as e:
                debug_info["errors"].append(f"Config error: {str(e)}")
            
            # Test connection if we have credentials
            if debug_info["api_credentials"] and bybit_crypto:
                try:
                    connection_test = asyncio.run(bybit_crypto.test_connection())
                    debug_info["connection_test"] = connection_test
                    if not connection_test:
                        debug_info["errors"].append("Connection test failed - check API credentials")
                except Exception as e:
                    debug_info["errors"].append(f"Connection test error: {str(e)}")
                
                # Test wallet balance with different account types
                if debug_info["connection_test"]:
                    account_types = ["UNIFIED", "CONTRACT", "SPOT"]
                    for account_type in account_types:
                        try:
                            result = asyncio.run(bybit_crypto.get_wallet_balance(account_type))
                            test_result = {
                                "account_type": account_type,
                                "success": result is not None,
                                "balance": result.total_balance_usd if result else 0,
                                "available": result.available_balance_usd if result else 0
                            }
                            debug_info["account_types_tested"].append(test_result)
                            if result and result.total_balance_usd > 0:
                                debug_info["wallet_balance_test"] = True
                        except Exception as e:
                            debug_info["account_types_tested"].append({
                                "account_type": account_type,
                                "success": False,
                                "error": str(e)
                            })
            
            return debug_info
            
        except Exception as exc:
            return {
                "error": f"Debug failed: {str(exc)}",
                "suggestion": "Check server logs for more details"
            }
    
    # ===== COMPREHENSIVE TRADING ANALYTICS ENDPOINTS =====
    
    @app.get("/analytics/{asset}/{mode}/reward-summary")
    def get_reward_summary(asset: str, mode: str):
        """Get comprehensive reward and performance summary."""
        valid_assets = ['crypto', 'futures', 'forex', 'forex_options']
        valid_modes = ['paper', 'live']
        
        if asset not in valid_assets:
            raise HTTPException(status_code=400, detail=f"Invalid asset. Must be one of: {valid_assets}")
        if mode not in valid_modes:
            raise HTTPException(status_code=400, detail=f"Invalid mode. Must be one of: {valid_modes}")
            
        try:
            if mode == "paper" and get_paper_trader:
                paper_trader = get_paper_trader(asset)
                stats = paper_trader.get_trading_stats()
                wallet_data = paper_trader.get_paper_wallet_data()
                
                # Calculate reward system metrics
                closed_trades = [t for t in paper_trader.trades if t.get('status') == 'closed']
                total_reward_points = sum(t.get('reward_points', 0) for t in closed_trades)
                avg_reward_per_trade = total_reward_points / len(closed_trades) if closed_trades else 0
                
                # Calculate consecutive stop loss hits
                consecutive_sl_count = 0
                for trade in reversed(closed_trades[-10:]):  # Check last 10 trades
                    if trade.get('exit_reason') == 'Stop Loss Hit':
                        consecutive_sl_count += 1
                    else:
                        break
                
                # Create reward summary from available data
                return {
                    "mode": "PAPER",
                    "asset_type": asset.upper(),
                    "overall_performance": {
                        "total_return_pct": stats.get('total_return', 0),
                        "net_pnl": stats.get('net_pnl', 0),
                        "current_balance": wallet_data.get('balance', 0),
                        "starting_balance": stats.get('starting_balance', 0)
                    },
                    "trading_metrics": {
                        "total_trades": stats.get('total_trades', 0),
                        "win_rate": stats.get('win_rate', 0),
                        "profit_factor": stats.get('profit_factor', 0),
                        "avg_win": stats.get('avg_win', 0),
                        "avg_loss": stats.get('avg_loss', 0),
                        "max_win": stats.get('max_win', 0),
                        "max_loss": stats.get('max_loss', 0)
                    },
                    "fees_and_costs": {
                        "total_penalties": stats.get('total_penalties', 0),
                        "violation_count": stats.get('violation_count', 0)
                    },
                    "reward_system_metrics": {
                        "total_reward_points": total_reward_points,
                        "avg_reward_per_trade": avg_reward_per_trade,
                        "consecutive_sl_hits": consecutive_sl_count
                    },
                    "recent_performance": wallet_data.get('history', [])[-10:]
                }
            elif mode == "live":
                # TODO: Implement live trading analytics
                return {
                    "mode": "LIVE",
                    "asset_type": asset.upper(),
                    "message": "Live trading analytics not yet implemented",
                    "overall_performance": {},
                    "trading_metrics": {},
                    "fees_and_costs": {},
                    "symbol_breakdown": {},
                    "recent_trades": []
                }
            else:
                raise HTTPException(status_code=404, detail="Paper trader not available")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to get reward summary: {exc}")
    
    @app.get("/analytics/{asset}/{mode}/detailed-trades")
    def get_detailed_trades(asset: str, mode: str, limit: int = 50):
        """Get detailed trade history with all information."""
        valid_assets = ['crypto', 'futures', 'forex', 'forex_options']
        valid_modes = ['paper', 'live']
        
        if asset not in valid_assets:
            raise HTTPException(status_code=400, detail=f"Invalid asset. Must be one of: {valid_assets}")
        if mode not in valid_modes:
            raise HTTPException(status_code=400, detail=f"Invalid mode. Must be one of: {valid_modes}")
        if limit > 500:  # Prevent excessive data requests
            limit = 500
            
        try:
            if mode == "paper" and get_paper_trader:
                paper_trader = get_paper_trader(asset)
                trades = paper_trader.get_detailed_trades(limit)
                return {
                    "trades": trades,
                    "total_count": len(paper_trader.trades),
                    "displayed_count": len(trades),
                    "asset": asset.upper(),
                    "mode": mode.upper()
                }
            elif mode == "live":
                # TODO: Implement live trading detailed trades
                return {
                    "trades": [],
                    "total_count": 0,
                    "displayed_count": 0,
                    "asset": asset.upper(),
                    "mode": mode.upper(),
                    "message": "Live trading trade history not yet implemented"
                }
            else:
                raise HTTPException(status_code=404, detail="Paper trader not available")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to get detailed trades: {exc}")
    
    @app.get("/analytics/{asset}/{mode}/symbol-performance")
    def get_symbol_performance(asset: str, mode: str):
        """Get performance breakdown by cryptocurrency symbols."""
        valid_assets = ['crypto', 'futures', 'forex', 'forex_options']
        valid_modes = ['paper', 'live']
        
        if asset not in valid_assets:
            raise HTTPException(status_code=400, detail=f"Invalid asset. Must be one of: {valid_assets}")
        if mode not in valid_modes:
            raise HTTPException(status_code=400, detail=f"Invalid mode. Must be one of: {valid_modes}")
            
        try:
            if mode == "paper" and get_paper_trader:
                paper_trader = get_paper_trader(asset)
                symbol_stats = paper_trader.get_performance_by_symbol()
                return {
                    "symbol_performance": symbol_stats,
                    "asset": asset.upper(),
                    "mode": mode.upper(),
                    "total_symbols": len(symbol_stats)
                }
            elif mode == "live":
                # TODO: Implement live trading symbol performance
                return {
                    "symbol_performance": {},
                    "asset": asset.upper(),
                    "mode": mode.upper(),
                    "total_symbols": 0,
                    "message": "Live trading symbol performance not yet implemented"
                }
            else:
                raise HTTPException(status_code=404, detail="Paper trader not available")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to get symbol performance: {exc}")
    
    @app.get("/analytics/{asset}/{mode}/trading-stats")
    def get_enhanced_trading_stats(asset: str, mode: str):
        """Get comprehensive trading statistics."""
        valid_assets = ['crypto', 'futures', 'forex', 'forex_options']
        valid_modes = ['paper', 'live']
        
        if asset not in valid_assets:
            raise HTTPException(status_code=400, detail=f"Invalid asset. Must be one of: {valid_assets}")
        if mode not in valid_modes:
            raise HTTPException(status_code=400, detail=f"Invalid mode. Must be one of: {valid_modes}")
            
        try:
            if mode == "paper" and get_paper_trader:
                paper_trader = get_paper_trader(asset)
                stats = paper_trader.get_trading_stats()
                return {
                    "statistics": stats,
                    "asset": asset.upper(),
                    "mode": mode.upper(),
                    "last_updated": datetime.now().isoformat()
                }
            elif mode == "live":
                # TODO: Implement live trading statistics
                return {
                    "statistics": {},
                    "asset": asset.upper(),
                    "mode": mode.upper(),
                    "message": "Live trading statistics not yet implemented"
                }
            else:
                raise HTTPException(status_code=404, detail="Paper trader not available")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to get trading stats: {exc}")
    
    @app.get("/analytics/overview")
    def get_analytics_overview():
        """Get overview of analytics across all assets and modes."""
        try:
            overview = {
                "paper_trading": {},
                "live_trading": {},
                "total_performance": {
                    "total_paper_pnl": 0,
                    "total_live_pnl": 0,
                    "total_trades": 0,
                    "best_performing_asset": "",
                    "worst_performing_asset": ""
                }
            }
            
            assets = ['crypto', 'futures', 'forex', 'forex_options']
            best_performance = -float('inf')
            worst_performance = float('inf')
            
            # Collect paper trading data
            for asset in assets:
                if get_paper_trader:
                    try:
                        paper_trader = get_paper_trader(asset)
                        stats = paper_trader.get_trading_stats()
                        
                        overview["paper_trading"][asset] = {
                            "balance": stats.get('current_balance', 0),
                            "total_return_pct": stats.get('total_return', 0),
                            "total_trades": stats.get('total_trades', 0),
                            "win_rate": stats.get('win_rate', 0),
                            "total_pnl": stats.get('total_pnl', 0)
                        }
                        
                        # Track best/worst performing
                        total_return = stats.get('total_return', 0)
                        if total_return > best_performance:
                            best_performance = total_return
                            overview["total_performance"]["best_performing_asset"] = f"{asset} (paper)"
                        if total_return < worst_performance:
                            worst_performance = total_return
                            overview["total_performance"]["worst_performing_asset"] = f"{asset} (paper)"
                        
                        # Add to totals
                        overview["total_performance"]["total_paper_pnl"] += stats.get('total_pnl', 0)
                        overview["total_performance"]["total_trades"] += stats.get('total_trades', 0)
                        
                    except Exception as e:
                        overview["paper_trading"][asset] = {"error": str(e)}
            
            # TODO: Add live trading data collection when implemented
            for asset in assets:
                overview["live_trading"][asset] = {
                    "status": "not_implemented",
                    "message": "Live trading analytics coming soon"
                }
            
            return overview
            
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to get analytics overview: {exc}")
    
    @app.get("/rewards/{asset}/recent")
    def get_recent_rewards(asset: str):
        """Get recent reward notifications for an asset."""
        valid_assets = ['crypto', 'futures', 'forex', 'forex_options']
        
        if asset not in valid_assets:
            raise HTTPException(status_code=400, detail=f"Invalid asset. Must be one of: {valid_assets}")
        
        try:
            if get_paper_trader:
                paper_trader = get_paper_trader(asset)
                recent_rewards = paper_trader.get_recent_rewards()
                return {
                    "asset": asset,
                    "recent_rewards": recent_rewards,
                    "count": len(recent_rewards),
                    "last_updated": datetime.now().isoformat()
                }
            else:
                return {
                    "asset": asset,
                    "recent_rewards": [],
                    "count": 0,
                    "message": "Paper trader not available"
                }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to get recent rewards: {exc}")
    
    # ===== STRATEGY DEVELOPMENT ENDPOINTS =====
    
    @app.get("/strategies/summary")
    def get_strategy_summary():
        """Get overall strategy development summary."""
        if not strategy_manager:
            return {
                "error": "Strategy development manager not available",
                "total_strategies": 0,
                "by_status": {},
                "by_asset": {}
            }
        
        try:
            return strategy_manager.get_strategy_summary()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to get strategy summary: {exc}")
    
    @app.get("/strategies/{asset}")
    def get_asset_strategies(asset: str):
        """Get strategies for a specific asset."""
        valid_assets = ['crypto', 'futures', 'forex', 'forex_options']
        if asset not in valid_assets:
            raise HTTPException(status_code=400, detail=f"Invalid asset. Must be one of: {valid_assets}")
        
        if not strategy_manager:
            return {
                "asset_type": asset,
                "strategies": {},
                "summary": {
                    "total": 0,
                    "developing": 0,
                    "pending_validation": 0,
                    "validated": 0,
                    "live": 0,
                    "rejected": 0
                }
            }
        
        try:
            return strategy_manager.get_strategies_by_asset(asset)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to get strategies for {asset}: {exc}")
    
    @app.post("/strategies/{strategy_id}/validate")
    async def validate_strategy(strategy_id: str):
        """Trigger validation for a strategy."""
        if not strategy_manager:
            raise HTTPException(status_code=503, detail="Strategy development manager not available")
        
        try:
            passed, reasons = await strategy_manager.validate_strategy(strategy_id)
            return {
                "strategy_id": strategy_id,
                "validation_passed": passed,
                "reasons": reasons,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to validate strategy {strategy_id}: {exc}")
    
    @app.post("/strategies/{strategy_id}/approve-live")
    def approve_strategy_for_live(strategy_id: str):
        """Approve a validated strategy for live testing."""
        if not strategy_manager:
            raise HTTPException(status_code=503, detail="Strategy development manager not available")
        
        try:
            success = strategy_manager.approve_for_live_testing(strategy_id)
            if success:
                return {
                    "strategy_id": strategy_id,
                    "approved": True,
                    "message": "Strategy approved for live testing",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                raise HTTPException(status_code=400, detail="Strategy not eligible for live approval")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to approve strategy {strategy_id}: {exc}")

    return app


app = create_app()
# reload server

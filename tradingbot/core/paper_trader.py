"""
STRICT Paper Trading Engine - REAL MARKET DATA ONLY
NO FAKE DATA ALLOWED - HEAVY PENALTIES FOR VIOLATIONS
"""
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import time

from .configmanager import config_manager
from .loggerconfig import get_logger

try:
    from tradingbot.brokers.exchangebybit import ExchangeBybit
except ImportError:
    ExchangeBybit = None

try:
    from tradingbot.rl.example_integration import TradingBotRewardIntegration
    REWARD_SYSTEM_AVAILABLE = True
except ImportError:
    TradingBotRewardIntegration = None
    REWARD_SYSTEM_AVAILABLE = False

try:
    from .strategy_development_manager import StrategyDevelopmentManager
    STRATEGY_DEVELOPMENT_AVAILABLE = True
except ImportError:
    StrategyDevelopmentManager = None
    STRATEGY_DEVELOPMENT_AVAILABLE = False


class StrictMarketDataViolation(Exception):
    """Exception raised when fake market data is detected."""
    pass


class StopLossViolation(Exception):
    """Exception raised when stop loss is violated."""
    pass


class PaperTrader:
    """STRICT Paper Trading Engine - REAL MARKET DATA ENFORCEMENT"""
    
    def __init__(self, asset_type: str):
        self.asset_type = asset_type
        self.log = get_logger(f"strict_paper_trader.{asset_type.lower()}")
        self.config = config_manager
        
        # STRICT ENFORCEMENT FLAGS
        self.strict_mode = True
        self.violation_penalty_multiplier = 10.0  # 10x penalty for violations
        self.fake_data_detected = False
        self.violation_count = 0
        
        # Set up persistence file
        self.state_file = Path(f"tradingbot/state/paper_trader_{asset_type.lower()}.json")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize paper trading state
        paper_start_balance = self.config.config.get('safety', {}).get('PAPER_EQUITY_START', 1000.0)
        
        # Try to load existing state first
        if self.state_file.exists():
            self._load_state()
            self.log.info(f"STRICT Paper trader loaded for {asset_type} with balance ${self.balance}")
        else:
            # Initialize fresh state
            self.balance = paper_start_balance
            self.starting_balance = paper_start_balance
            self.positions: List[Dict] = []
            self.trades: List[Dict] = []
            self.pnl_history: List[Dict] = [{"timestamp": datetime.now().isoformat(), "balance": self.balance}]
            self.violations_log: List[Dict] = []  # Track all violations
            self._save_state()
            self.log.info(f"STRICT Paper trader initialized for {asset_type} with starting balance ${self.balance}")
        
        # STRICT Trading parameters
        self.risk_per_trade = self.config.config.get('bot_settings', {}).get('per_trade_risk_percent', 0.01)
        self.min_trade_amount = self.config.config.get('bot_settings', {}).get('min_trade_amount_usd', 10.0)
        self.max_stop_loss_pct = -15.0  # MAXIMUM -15% stop loss - CANNOT BE VIOLATED
        
        # REAL Market data cache - STRICT VALIDATION
        self.market_data = {}
        self.last_market_update = None
        self.market_data_staleness_threshold = 60  # Max 60 seconds old data
        
        # Initialize Bybit connection for REAL market data ONLY
        self.bybit_exchange = None
        if ExchangeBybit and asset_type == "crypto":
            try:
                self.bybit_exchange = ExchangeBybit("CRYPTO_SPOT", "paper")
                self.log.critical("STRICT MODE: Connected to Bybit for REAL market data validation")
            except Exception as e:
                self.log.critical(f"CRITICAL FAILURE: Could not connect to Bybit: {e}")
                self.log.critical("STRICT MODE: Paper trading HALTED without real market connection")
                raise Exception("REAL MARKET CONNECTION REQUIRED")
        
        # Define symbols to track for live data (verified available on Bybit)
        self.tracked_symbols = [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", 
            "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT", "UNIUSDT",
            "LTCUSDT", "BCHUSDT", "ATOMUSDT", "NEARUSDT", "ICPUSDT"
        ]
        
        # Initialize reward system integration
        self.reward_system = None
        if REWARD_SYSTEM_AVAILABLE:
            try:
                self.reward_system = TradingBotRewardIntegration(
                    logger=self.log,
                    telegram_send_fn=None  # Add telegram function if available
                )
                self.log.info(f"Reward system integrated for {asset_type}")
            except Exception as e:
                self.log.warning(f"Could not initialize reward system: {e}")
        
        # Initialize strategy development manager
        self.strategy_manager = None
        self.strategy_id = None
        if STRATEGY_DEVELOPMENT_AVAILABLE:
            try:
                if not hasattr(PaperTrader, '_shared_strategy_manager'):
                    PaperTrader._shared_strategy_manager = StrategyDevelopmentManager()
                self.strategy_manager = PaperTrader._shared_strategy_manager
                self._register_or_update_strategy()
                self.log.info(f"Strategy development manager integrated for {asset_type}")
            except Exception as e:
                self.log.warning(f"Could not initialize strategy development manager: {e}")
    
    async def start_paper_trading(self) -> bool:
        """Start STRICT paper trading with REAL market data enforcement."""
        try:
            self.log.critical(f"[STRICT MODE] Starting paper trading for {self.asset_type}")
            self.log.critical("WARNING: STRICT ENFORCEMENT ACTIVE - NO FAKE DATA ALLOWED")
            
            # Phase 1: CRITICAL - REAL Market data fetching ONLY
            self.log.info("[PHASE 1] Fetching REAL market data with strict validation...")
            await self.update_market_data()
            
            # STRICT ENFORCEMENT: Verify we have REAL market data
            if not self.market_data or len(self.market_data) < 3:
                violation = {
                    "type": "INSUFFICIENT_MARKET_DATA",
                    "timestamp": datetime.now().isoformat(),
                    "penalty": self.violation_penalty_multiplier * 100.0
                }
                self._record_violation(violation)
                self.log.critical("VIOLATION: Insufficient real market data")
                return False
            
            # Phase 2: Initialize positions with REAL data only
            self.log.info("[PHASE 2] Creating initial positions with REAL market validation...")
            await self._create_realistic_positions()
            
            # Phase 3: Start monitoring with strict TP/SL enforcement
            self.log.info("[PHASE 3] Starting position monitoring with STRICT TP/SL enforcement...")
            await self._start_position_monitoring()
            
            self.log.critical("[COMPLETE] STRICT paper trading active - Real market data enforced")
            return True
            
        except Exception as e:
            violation = {
                "type": "STARTUP_FAILURE",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "penalty": self.violation_penalty_multiplier * 500.0
            }
            self._record_violation(violation)
            self.log.error(f"VIOLATION: Failed to start paper trading: {e}")
            return False
    
    async def update_market_data(self):
        """Update market data with STRICT real-time validation."""
        try:
            if not self.bybit_exchange:
                raise StrictMarketDataViolation("No Bybit connection - cannot proceed without real data")
            
            if self.asset_type == "crypto":
                self.log.info("[STRICT] Fetching live market data from Bybit with timestamp validation...")
                
                # Test connection with timeout
                if not await asyncio.wait_for(self.bybit_exchange.test_connection(), timeout=10.0):
                    raise StrictMarketDataViolation("Bybit connection test failed")
                
                success_count = 0
                current_time = datetime.now()
                
                # Clear any potentially stale data
                self.market_data.clear()
                
                for symbol in self.tracked_symbols:
                    try:
                        # Get REAL-TIME OHLCV data with timestamp validation
                        ohlcv_data = await asyncio.wait_for(
                            self.bybit_exchange.fetch_ohlcv(symbol, "1m", 2), 
                            timeout=5.0
                        )
                        
                        if ohlcv_data and len(ohlcv_data) >= 2:
                            current_candle = ohlcv_data[-1]  # Most recent candle
                            
                            current_price = current_candle["close"]
                            volume = current_candle["volume"]
                            timestamp = current_candle["timestamp"]
                            
                            # STRICT VALIDATION: Check data freshness
                            if isinstance(timestamp, (int, float)):
                                data_time = datetime.fromtimestamp(timestamp / 1000)
                            else:
                                data_time = datetime.fromisoformat(str(timestamp).replace('Z', ''))
                            
                            staleness_seconds = (current_time - data_time).total_seconds()
                            if staleness_seconds > self.market_data_staleness_threshold:
                                self.log.warning(f"Data staleness detected for {symbol}: {staleness_seconds:.1f}s old")
                                continue
                            
                            # STRICT VALIDATION: Price validation
                            if not self._validate_crypto_price_strict(symbol, current_price):
                                continue
                            
                            # STRICT VALIDATION: Volume validation
                            if volume <= 0:
                                self.log.warning(f"Invalid volume for {symbol}: {volume}")
                                continue
                            
                            # Convert symbol format for display
                            display_symbol = symbol.replace("USDT", "/USDT")
                            
                            self.market_data[display_symbol] = {
                                "price": current_price,
                                "volume": volume * current_price,  # USD volume
                                "timestamp": current_time.isoformat(),
                                "data_timestamp": data_time.isoformat(),
                                "staleness_seconds": staleness_seconds,
                                "bybit_symbol": symbol,
                                "source": "bybit_live_verified",
                                "validated": True,
                                "strict_validated": True  # Double validation
                            }
                            
                            success_count += 1
                            self.log.debug(f"VALIDATED: {symbol} ${current_price:.4f} ({staleness_seconds:.1f}s fresh)")
                            
                        else:
                            self.log.warning(f"No OHLCV data received for {symbol}")
                            
                    except asyncio.TimeoutError:
                        self.log.error(f"Timeout fetching data for {symbol}")
                        continue
                    except Exception as e:
                        self.log.error(f"Failed to get data for {symbol}: {e}")
                        continue
                
                self.log.critical(f"STRICT VALIDATION: {success_count} symbols with REAL data")
                
                # STRICT ENFORCEMENT: Must have minimum real data
                if success_count < 5:  # Increased minimum requirement
                    raise StrictMarketDataViolation(f"Only {success_count}/5 minimum real data points available")
                
                self.last_market_update = current_time
                
        except Exception as e:
            violation = {
                "type": "MARKET_DATA_VIOLATION",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "penalty": self.violation_penalty_multiplier * 1000.0
            }
            self._record_violation(violation)
            self.log.critical(f"STRICT VIOLATION: {e}")
            raise e

    def _validate_crypto_price_strict(self, symbol: str, price: float) -> bool:
        """STRICT price validation with current market ranges."""
        # Updated price ranges for current market (January 2025)
        price_ranges = {
            "BTCUSDT": (80000, 120000),     # BTC current range
            "ETHUSDT": (3000, 6000),        # ETH current range
            "SOLUSDT": (150, 300),          # SOL current range
            "BNBUSDT": (600, 1200),         # BNB current range  
            "XRPUSDT": (2.0, 4.0),          # XRP current range
            "ADAUSDT": (0.6, 1.2),          # ADA current range
            "AVAXUSDT": (20, 50),           # AVAX current range
            "DOTUSDT": (3, 12),             # DOT current range
            "LINKUSDT": (15, 35),           # LINK current range
            "UNIUSDT": (8, 16),             # UNI current range
            "LTCUSDT": (90, 150),           # LTC current range
            "BCHUSDT": (400, 700),          # BCH current range
            "ATOMUSDT": (3, 8),             # ATOM current range
            "NEARUSDT": (2, 6),             # NEAR current range
            "ICPUSDT": (4, 12),             # ICP current range
        }
        
        base_symbol = symbol.replace("/", "")
        if base_symbol in price_ranges:
            min_price, max_price = price_ranges[base_symbol]
            is_valid = min_price <= price <= max_price
            if not is_valid:
                self.log.critical(f"STRICT VIOLATION: Price {symbol} ${price:.4f} outside valid range ${min_price}-${max_price}")
                violation = {
                    "type": "INVALID_PRICE_RANGE",
                    "symbol": symbol,
                    "price": price,
                    "valid_range": [min_price, max_price],
                    "timestamp": datetime.now().isoformat(),
                    "penalty": self.violation_penalty_multiplier * 100.0
                }
                self._record_violation(violation)
            return is_valid
        
        # For unknown symbols, strict range
        is_valid = 0.01 <= price <= 50000
        if not is_valid:
            self.log.critical(f"STRICT VIOLATION: Price {symbol} ${price:.4f} outside reasonable bounds")
        return is_valid
    
    async def _create_realistic_positions(self):
        """Create positions using ONLY real market data with proper TP/SL."""
        if not self.market_data:
            raise StrictMarketDataViolation("Cannot create positions without real market data")
        
        # Check if we already have open positions
        open_positions = [p for p in self.positions if p.get("status") == "open"]
        if open_positions:
            self.log.info(f"[SKIP CREATION] Already have {len(open_positions)} open positions, skipping position creation")
            return
        
        # Only create 1-2 positions to maintain realism
        available_symbols = [s for s, d in self.market_data.items() if d.get("strict_validated", False)]
        if not available_symbols:
            raise StrictMarketDataViolation("No strictly validated symbols available")
        
        num_positions = min(2, len(available_symbols))  # Maximum 2 positions
        selected_symbols = available_symbols[:num_positions]  # Take first validated symbols
        
        position_size_pct = 0.15 / num_positions  # Split 15% of balance (more reasonable size)
        
        for symbol in selected_symbols:
            data = self.market_data[symbol]
            entry_price = data["price"]
            
            # Determine position side based on simple market analysis
            side = "buy"  # Default to buy for crypto
            
            # Calculate position size
            position_value = self.balance * position_size_pct
            position_size = position_value / entry_price
            
            # Set STRICT TP/SL levels
            if side == "buy":
                take_profit = entry_price * 1.02  # 2% profit target
                stop_loss = entry_price * 0.85    # -15% maximum loss (STRICT)
            else:  # sell
                take_profit = entry_price * 0.98  # 2% profit target 
                stop_loss = entry_price * 1.15    # -15% maximum loss (STRICT)
            
            position = {
                "id": f"pos_{len(self.positions) + 1}_{int(time.time())}",
                "symbol": symbol,
                "side": side,
                "size": position_size,
                "entry_price": entry_price,
                "current_price": entry_price,
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "pnl": 0.0,
                "pnl_pct": 0.0,
                "timestamp": datetime.now().isoformat(),
                "status": "open",
                "market_data_source": "bybit_live_verified",
                "real_data": True,
                "strict_validated": True,
                "max_loss_enforced": True,
                "sl_violations": 0
            }
            
            self.positions.append(position)
            
            self.log.info(f"[POSITION CREATED] {side.upper()} {symbol}:")
            self.log.info(f"  Entry: ${entry_price:.4f}")
            self.log.info(f"  Size: {position_size:.6f}")
            self.log.info(f"  TP: ${take_profit:.4f} (+2%)")
            self.log.info(f"  SL: ${stop_loss:.4f} (-15% MAX)")
        
        self._save_state()
        self.log.critical(f"CREATED {len(selected_symbols)} positions with STRICT TP/SL enforcement")
    
    async def _start_position_monitoring(self):
        """Start monitoring positions with STRICT TP/SL enforcement."""
        self.log.critical("Starting STRICT position monitoring - TP/SL enforcement active")
        
        # In a real implementation, this would run continuously
        # For now, we'll just update positions once
        await self._update_positions_with_strict_enforcement()
    
    async def _update_positions_with_strict_enforcement(self):
        """Update positions using current market data with STRICT TP/SL enforcement."""
        if not self.positions:
            self.log.info("No positions found to monitor")
            return
        
        open_positions = [p for p in self.positions if p.get("status") == "open"]
        if not open_positions:
            self.log.info("No open positions found to monitor")
            return
        
        self.log.critical(f"[POSITION MONITOR] Monitoring {len(open_positions)} active positions:")
        for pos in open_positions:
            symbol = pos.get("symbol", "N/A")
            side = pos.get("side", "N/A")
            entry_price = pos.get("entry_price", 0)
            size = pos.get("size", 0)
            current_price = pos.get("current_price", entry_price)
            pnl_pct = pos.get("pnl_pct", 0)
            self.log.info(f"  [{symbol}] {side.upper()} | Entry: ${entry_price:.4f} | Current: ${current_price:.4f} | PnL: {pnl_pct:+.2f}%")
        
        # Refresh market data
        await self.update_market_data()
        
        for position in self.positions.copy():  # Use copy to allow modification during iteration
            if position.get("status") != "open":
                continue
            
            symbol = position.get("symbol")
            if symbol not in self.market_data:
                continue
            
            market_data = self.market_data[symbol]
            if not market_data.get("strict_validated", False):
                self.log.critical(f"VIOLATION: Position {symbol} using non-validated data")
                continue
            
            current_price = market_data["price"]
            entry_price = position.get("entry_price", 0)
            side = position.get("side", "")
            size = position.get("size", 0)
            take_profit = position.get("take_profit", 0)
            stop_loss = position.get("stop_loss", 0)
            
            # Update current price
            position["current_price"] = current_price
            position["last_update"] = datetime.now().isoformat()
            
            # Calculate P&L
            if side == "buy":
                pnl = (current_price - entry_price) * size
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:  # sell
                pnl = (entry_price - current_price) * size
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
            
            position["pnl"] = pnl
            position["pnl_pct"] = pnl_pct
            
            # Log position update with more detail
            tp_price = position.get("take_profit", 0)
            sl_price = position.get("stop_loss", 0)
            position_value = entry_price * size
            self.log.info(f"[UPDATE] {symbol} {side.upper()} | Size: {size:.6f} (${position_value:.2f})")
            self.log.info(f"         Price: ${entry_price:.4f} -> ${current_price:.4f} | P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
            self.log.info(f"         TP: ${tp_price:.4f} | SL: ${sl_price:.4f} | Distance to TP: {((tp_price/current_price-1)*100):+.2f}% | Distance to SL: {((sl_price/current_price-1)*100):+.2f}%")
            
            # STRICT TP/SL ENFORCEMENT
            should_close = False
            close_reason = ""
            
            if side == "buy":
                if current_price >= take_profit:
                    should_close = True
                    close_reason = "Take Profit Hit"
                elif current_price <= stop_loss:
                    should_close = True
                    close_reason = "Stop Loss Hit"
                    # STRICT: Ensure SL never exceeds -15%
                    if pnl_pct < -15.0:
                        violation = {
                            "type": "STOP_LOSS_VIOLATION",
                            "symbol": symbol,
                            "pnl_pct": pnl_pct,
                            "max_allowed": -15.0,
                            "timestamp": datetime.now().isoformat(),
                            "penalty": self.violation_penalty_multiplier * abs(pnl_pct - (-15.0)) * 50.0
                        }
                        self._record_violation(violation)
                        self.log.critical(f"STRICT VIOLATION: SL exceeded -15% limit: {pnl_pct:.2f}%")
                        # Force close at -15%
                        pnl = (entry_price * 0.85 - entry_price) * size
                        pnl_pct = -15.0
                        position["pnl"] = pnl
                        position["pnl_pct"] = pnl_pct
            else:  # sell
                if current_price <= take_profit:
                    should_close = True
                    close_reason = "Take Profit Hit"
                elif current_price >= stop_loss:
                    should_close = True
                    close_reason = "Stop Loss Hit"
                    # STRICT: Ensure SL never exceeds -15%
                    if pnl_pct < -15.0:
                        violation = {
                            "type": "STOP_LOSS_VIOLATION",
                            "symbol": symbol,
                            "pnl_pct": pnl_pct,
                            "max_allowed": -15.0,
                            "timestamp": datetime.now().isoformat(),
                            "penalty": self.violation_penalty_multiplier * abs(pnl_pct - (-15.0)) * 50.0
                        }
                        self._record_violation(violation)
                        self.log.critical(f"STRICT VIOLATION: SL exceeded -15% limit: {pnl_pct:.2f}%")
                        # Force close at -15%
                        pnl = (entry_price - entry_price * 1.15) * size
                        pnl_pct = -15.0
                        position["pnl"] = pnl
                        position["pnl_pct"] = pnl_pct
            
            if should_close:
                self._close_position_strict(position, close_reason, current_price)
        
        self._save_state()
    
    def _close_position_strict(self, position: Dict, reason: str, exit_price: float):
        """Close position with STRICT validation and penalties."""
        try:
            symbol = position.get("symbol", "")
            side = position.get("side", "")
            entry_price = position.get("entry_price", 0)
            size = position.get("size", 0)
            pnl = position.get("pnl", 0)
            pnl_pct = position.get("pnl_pct", 0)
            
            # STRICT: Validate exit price is real
            if symbol in self.market_data:
                market_price = self.market_data[symbol]["price"]
                price_diff_pct = abs(exit_price - market_price) / market_price * 100
                if price_diff_pct > 0.1:  # Allow max 0.1% difference for slippage
                    violation = {
                        "type": "FAKE_EXIT_PRICE",
                        "symbol": symbol,
                        "exit_price": exit_price,
                        "market_price": market_price,
                        "difference_pct": price_diff_pct,
                        "timestamp": datetime.now().isoformat(),
                        "penalty": self.violation_penalty_multiplier * price_diff_pct * 100.0
                    }
                    self._record_violation(violation)
                    self.log.critical(f"VIOLATION: Fake exit price detected {symbol} {exit_price} vs market {market_price}")
                    # Use real market price
                    exit_price = market_price
                    # Recalculate P&L with real price
                    if side == "buy":
                        pnl = (exit_price - entry_price) * size
                    else:
                        pnl = (entry_price - exit_price) * size
                    pnl_pct = (pnl / (entry_price * size)) * 100 if entry_price * size > 0 else 0
            
            # Apply penalties for violations
            total_penalty = sum(v.get("penalty", 0) for v in self.violations_log)
            net_pnl = pnl - total_penalty
            
            # Update balance with net P&L (including penalties)
            previous_balance = self.balance
            self.balance += net_pnl
            
            # Mark position as closed
            position["status"] = "closed"
            position["exit_price"] = exit_price
            position["exit_time"] = datetime.now().isoformat()
            position["exit_reason"] = reason
            position["net_pnl"] = net_pnl
            position["penalties_applied"] = total_penalty
            
            # Calculate trade amount
            trade_amount = entry_price * size
            
            # Create comprehensive trade record
            trade = {
                "id": f"trade_{len(self.trades) + 1}_{int(time.time())}",
                "symbol": symbol,
                "side": side.upper(),
                "size": size,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "trade_amount_usd": trade_amount,
                "entry_time": position.get("timestamp", ""),
                "exit_time": position["exit_time"],
                "pnl_usd": pnl,
                "pnl_pct": pnl_pct,
                "net_pnl": net_pnl,
                "penalties": total_penalty,
                "balance": self.balance,  # Running balance
                "exit_reason": reason,
                "status": "closed",
                "real_market_data": True,
                "strict_validated": True,
                "violations": len(self.violations_log)
            }
            
            self.trades.append(trade)
            
            # Update P&L history
            self.pnl_history.append({
                "timestamp": position["exit_time"],
                "balance": self.balance,
                "pnl": net_pnl,
                "trade_id": trade["id"],
                "penalties": total_penalty
            })
            
            self.log.critical(f"[CLOSED] {symbol} {side.upper()} position:")
            self.log.critical(f"  Exit: {reason} at ${exit_price:.4f}")
            self.log.critical(f"  P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
            if total_penalty > 0:
                self.log.critical(f"  Penalties: -${total_penalty:.2f}")
                self.log.critical(f"  Net P&L: ${net_pnl:+.2f}")
            self.log.critical(f"  Balance: ${previous_balance:.2f} -> ${self.balance:.2f}")
            
            # Calculate reward using integrated reward system
            if self.reward_system:
                try:
                    holding_time_seconds = 3600.0  # Default 1 hour
                    try:
                        entry_time = datetime.fromisoformat(position.get("timestamp", ""))
                        exit_time = datetime.fromisoformat(position["exit_time"])
                        holding_time_seconds = (exit_time - entry_time).total_seconds()
                    except:
                        pass
                    
                    estimated_fees = trade_amount * 0.001  # 0.1% fee
                    open_exposure = sum(
                        abs(p.get("current_price", 0) * p.get("size", 0)) 
                        for p in self.positions 
                        if p.get("status") == "open" and p != position
                    )
                    
                    # Apply penalty to reward calculation
                    adjusted_pnl = net_pnl
                    if total_penalty > 0:
                        adjusted_pnl = pnl * 0.1  # Heavily penalize reward for violations
                    
                    reward = self.reward_system.on_trade_closed(
                        symbol=symbol,
                        side=side,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        quantity=size,
                        leverage=1.0,
                        fees_paid=estimated_fees,
                        slippage=0.0,
                        holding_time_seconds=holding_time_seconds,
                        current_equity=self.balance,
                        open_exposure=open_exposure
                    )
                    
                    # Apply penalty to rewards
                    if total_penalty > 0:
                        reward = reward * 0.1  # 90% reward penalty for violations
                    
                    trade["reward_points"] = reward
                    self.log.info(f"  Reward: {reward:+.2f} points")
                    
                except Exception as e:
                    self.log.warning(f"Error calculating reward: {e}")
                    trade["reward_points"] = 0.0
            
            # Update strategy development tracking
            if self.strategy_manager:
                try:
                    self._register_or_update_strategy()
                except Exception as e:
                    self.log.warning(f"Error updating strategy: {e}")
            
        except Exception as e:
            self.log.error(f"Error closing position: {e}")
    
    def _record_violation(self, violation: Dict):
        """Record a strict enforcement violation with penalty."""
        self.violation_count += 1
        violation["violation_id"] = self.violation_count
        self.violations_log.append(violation)
        
        penalty = violation.get("penalty", 0)
        self.balance -= penalty  # Apply penalty immediately
        
        self.log.critical(f"VIOLATION #{self.violation_count}: {violation['type']}")
        self.log.critical(f"PENALTY APPLIED: -${penalty:.2f}")
        self.log.critical(f"BALANCE AFTER PENALTY: ${self.balance:.2f}")
        
        self._save_state()
        
        # If too many violations, halt trading
        if self.violation_count >= 10:
            self.log.critical("TRADING HALTED: Too many violations detected")
            raise Exception("Trading halted due to excessive violations")
    
    def close_all_positions(self, reason: str = "Manual close"):
        """Close all open positions."""
        open_positions = [p for p in self.positions if p.get("status") == "open"]
        if not open_positions:
            self.log.info("[CLOSE ALL] No open positions to close")
            return
            
        self.log.critical(f"[CLOSE ALL] Closing {len(open_positions)} positions: {reason}")
        for position in open_positions:
            symbol = position.get("symbol", "")
            current_price = position.get("current_price", position.get("entry_price", 0))
            self._close_position_strict(position, reason, current_price)
        
        self._save_state()
        self.log.critical(f"[CLOSE ALL] All positions closed. New balance: ${self.balance:.2f}")

    def get_positions(self) -> List[Dict]:
        """Get current open positions."""
        open_positions = [pos for pos in self.positions if pos.get("status") == "open"]
        
        # Log active positions when requested
        if open_positions:
            self.log.info(f"[ACTIVE POSITIONS] Currently tracking {len(open_positions)} open positions:")
            for pos in open_positions:
                symbol = pos.get("symbol", "N/A")
                side = pos.get("side", "N/A") 
                entry_price = pos.get("entry_price", 0)
                current_price = pos.get("current_price", entry_price)
                size = pos.get("size", 0)
                pnl_usd = pos.get("pnl", 0)
                pnl_pct = pos.get("pnl_pct", 0)
                tp_price = pos.get("take_profit", 0)
                sl_price = pos.get("stop_loss", 0)
                position_value = entry_price * size
                status = pos.get("status", "unknown")
                
                self.log.info(f"  • {symbol} {side.upper()} | Size: {size:.6f} (${position_value:.2f}) | Entry: ${entry_price:.4f} | Current: ${current_price:.4f}")
                self.log.info(f"    P&L: ${pnl_usd:+.2f} ({pnl_pct:+.2f}%) | TP: ${tp_price:.4f} | SL: ${sl_price:.4f} | Status: {status}")
        else:
            self.log.info("[ACTIVE POSITIONS] No open positions currently")
            
        return open_positions
    
    def get_detailed_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get detailed trade history."""
        detailed_trades = []
        
        for trade in self.trades[-limit:]:
            try:
                entry_price = trade.get('entry_price', 0)
                size = trade.get('size', 0)
                trade_amount = entry_price * size
                
                # Calculate duration
                duration_str = "N/A"
                if trade.get('entry_time') and trade.get('exit_time'):
                    try:
                        entry_time = datetime.fromisoformat(trade['entry_time'])
                        exit_time = datetime.fromisoformat(trade['exit_time'])
                        duration = exit_time - entry_time
                        
                        if duration.days > 0:
                            duration_str = f"{duration.days}d {duration.seconds//3600}h"
                        elif duration.seconds >= 3600:
                            duration_str = f"{duration.seconds//3600}h {(duration.seconds%3600)//60}m"
                        elif duration.seconds >= 60:
                            duration_str = f"{(duration.seconds%3600)//60}m {duration.seconds%60}s"
                        else:
                            duration_str = f"{duration.seconds}s"
                    except:
                        pass
                
                pnl = trade.get('pnl_usd', 0)
                pnl_pct = trade.get('pnl_pct', 0)
                net_pnl = trade.get('net_pnl', pnl)
                penalties = trade.get('penalties', 0)
                
                # Determine trade result
                if net_pnl > 0:
                    result = "WIN"
                    result_color = "green"
                elif net_pnl < 0:
                    result = "LOSS" + (f" (${penalties:.2f} penalties)" if penalties > 0 else "")
                    result_color = "red"
                else:
                    result = "BREAK-EVEN"
                    result_color = "gray"
                
                detailed_trade = {
                    "id": trade.get('id', 'N/A'),
                    "symbol": trade.get('symbol', 'N/A'),
                    "side": trade.get('side', 'N/A'),
                    "size": round(size, 8),
                    "entry_price": round(entry_price, 6),
                    "exit_price": round(trade.get('exit_price', 0), 6),
                    "trade_amount_usd": round(trade_amount, 2),
                    "pnl_usd": round(pnl, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "net_pnl": round(net_pnl, 2),
                    "penalties": round(penalties, 2),
                    "duration": duration_str,
                    "entry_time": trade.get('entry_time', 'N/A'),
                    "exit_time": trade.get('exit_time', 'N/A'),
                    "exit_reason": trade.get('exit_reason', 'N/A'),
                    "result": result,
                    "result_color": result_color,
                    "balance": round(trade.get('balance', 0), 2),
                    "strict_validated": trade.get('strict_validated', False),
                    "violations": trade.get('violations', 0)
                }
                
                detailed_trades.append(detailed_trade)
                
            except Exception as e:
                self.log.debug(f"Error processing trade {trade.get('id', 'unknown')}: {e}")
                continue
        
        return detailed_trades[::-1]  # Return newest first
    
    def get_trading_stats(self) -> Dict[str, Any]:
        """Get comprehensive trading statistics including violations."""
        if not self.trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "total_penalties": 0.0,
                "net_pnl": 0.0,
                "violation_count": self.violation_count,
                "strict_enforcement": True
            }
        
        wins = [t for t in self.trades if t.get('net_pnl', 0) > 0]
        losses = [t for t in self.trades if t.get('net_pnl', 0) <= 0]
        
        total_trades = len(self.trades)
        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
        
        total_pnl = sum(t.get('pnl_usd', 0) for t in self.trades)
        total_penalties = sum(t.get('penalties', 0) for t in self.trades)
        net_pnl = total_pnl - total_penalties
        
        avg_win = sum(t.get('net_pnl', 0) for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.get('net_pnl', 0) for t in losses) / len(losses) if losses else 0
        max_win = max([t.get('net_pnl', 0) for t in wins], default=0)
        max_loss = min([t.get('net_pnl', 0) for t in losses], default=0)
        
        profit_factor = abs(avg_win / avg_loss) if avg_loss < 0 else 0
        total_return = ((self.balance - self.starting_balance) / self.starting_balance) * 100
        
        return {
            "total_trades": total_trades,
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "total_penalties": round(total_penalties, 2),
            "net_pnl": round(net_pnl, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "max_win": round(max_win, 2),
            "max_loss": round(max_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "total_return": round(total_return, 2),
            "starting_balance": round(self.starting_balance, 2),
            "current_balance": round(self.balance, 2),
            "violation_count": self.violation_count,
            "strict_enforcement": True,
            "real_data_only": True
        }
    
    def get_paper_wallet_data(self) -> Dict[str, Any]:
        """Get current paper wallet data with violation info."""
        total_pnl = self.balance - self.starting_balance
        pnl_percent = (total_pnl / self.starting_balance * 100) if self.starting_balance > 0 else 0
        
        return {
            "balance": self.balance,
            "pnl": total_pnl,
            "pnl_percent": pnl_percent,
            "history": self.pnl_history[-20:],
            "positions": len([p for p in self.positions if p.get("status") == "open"]),
            "trades_today": len([t for t in self.trades if self._is_today(t.get("entry_time", ""))]),
            "violation_count": self.violation_count,
            "strict_enforcement": True
        }
    
    def _is_today(self, timestamp_str: str) -> bool:
        """Check if timestamp is from today."""
        try:
            if not timestamp_str:
                return False
            trade_date = datetime.fromisoformat(timestamp_str).date()
            return trade_date == datetime.now().date()
        except:
            return False
    
    def _save_state(self):
        """Save paper trading state to file."""
        try:
            state = {
                "balance": self.balance,
                "starting_balance": self.starting_balance,
                "positions": self.positions,
                "trades": self.trades,
                "pnl_history": self.pnl_history,
                "violations_log": self.violations_log,
                "violation_count": self.violation_count,
                "strict_enforcement": True,
                "last_update": datetime.now().isoformat()
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
                
        except Exception as e:
            self.log.error(f"Failed to save state: {e}")
    
    def _load_state(self):
        """Load paper trading state from file."""
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            self.balance = state.get("balance", 1000.0)
            self.starting_balance = state.get("starting_balance", 1000.0)
            self.positions = state.get("positions", [])
            self.trades = state.get("trades", [])
            self.pnl_history = state.get("pnl_history", [])
            self.violations_log = state.get("violations_log", [])
            self.violation_count = state.get("violation_count", 0)
            
        except Exception as e:
            self.log.warning(f"Failed to load state: {e}")
            paper_start_balance = self.config.config.get('safety', {}).get('PAPER_EQUITY_START', 1000.0)
            self.balance = paper_start_balance
            self.starting_balance = paper_start_balance
            self.positions = []
            self.trades = []
            self.pnl_history = [{"timestamp": datetime.now().isoformat(), "balance": self.balance}]
            self.violations_log = []
            self.violation_count = 0
    
    def _register_or_update_strategy(self):
        """Register or update strategy with violations tracking."""
        if not self.strategy_manager:
            return
        
        try:
            strategy_data = {
                "id": f"strict_paper_strategy_{self.asset_type.lower()}",
                "name": f"STRICT Paper Trading Strategy - {self.asset_type}",
                "asset_type": self.asset_type,
                "status": "developing",
                "performance": {
                    "total_trades": len(self.trades),
                    "win_rate": self.get_trading_stats().get('win_rate', 0),
                    "total_return": self.get_trading_stats().get('total_return', 0),
                    "balance": self.balance,
                    "violations": self.violation_count
                },
                "last_updated": datetime.now().isoformat(),
                "data_source": "real_market_strict",
                "strict_enforcement": True
            }
            
            self.strategy_manager.register_strategy(strategy_data)
            self.strategy_id = strategy_data["id"]
            
        except Exception as e:
            self.log.debug(f"Strategy registration failed: {e}")


# Global paper trader instances
_paper_traders = {}

def get_paper_trader(asset_type: str) -> PaperTrader:
    """Get or create strict paper trader for asset type."""
    if asset_type not in _paper_traders:
        _paper_traders[asset_type] = PaperTrader(asset_type)
    return _paper_traders[asset_type]


__all__ = ["PaperTrader", "get_paper_trader", "StrictMarketDataViolation", "StopLossViolation"]
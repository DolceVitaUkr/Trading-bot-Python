"""
Trading Engine - Executes paper trading strategies for all assets.
"""
import asyncio
import random
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

from .paper_trader import get_paper_trader
from .loggerconfig import get_logger
from .runtime_controller import RuntimeController
from .activity_logger import log_activity
from ..strategies import IndicatorStrategy, create_strategy_variants
from ..brokers.exchangebybit import ExchangeBybit
from ..brokers.connectibkrapi import IBKRConnectionManager


class TradingEngine:
    """Main trading engine that runs strategies for all assets."""
    
    def __init__(self):
        self.log = get_logger("trading_engine")
        self.runtime = RuntimeController()
        
        # Get shared paper traders for each asset
        self.paper_traders = {
            'crypto': get_paper_trader('crypto'),
            'futures': get_paper_trader('futures'),
            'forex': get_paper_trader('forex'),
            'forex_options': get_paper_trader('forex_options')
        }
        
        # Trading loops
        self.trading_tasks = {}
        self.is_running = False
        
        # Initialize strategies for each asset
        self.strategies = {}
        self._initialize_strategies()
        
        # Initialize broker connections
        self.brokers = {}
        self._initialize_brokers()
        
        # Strategy parameters
        self.strategy_params = {
            'crypto': {
                'symbols': ['BTCUSDT', 'ETHUSDT'],
                'check_interval': 10,  # seconds (reduced for testing)
                'position_size': 0.01,  # 1% of balance
                'take_profit': 0.02,    # 2%
                'stop_loss': -0.01      # -1%
            },
            'futures': {
                'symbols': ['BTCUSDT'],
                'check_interval': 45,
                'position_size': 0.02,
                'take_profit': 0.03,
                'stop_loss': -0.015
            },
            'forex': {
                'symbols': ['EURUSD', 'GBPUSD'],
                'check_interval': 60,
                'position_size': 0.02,
                'take_profit': 0.01,
                'stop_loss': -0.005
            },
            'forex_options': {
                'symbols': ['EURUSD'],
                'check_interval': 90,
                'position_size': 0.01,
                'take_profit': 0.05,
                'stop_loss': -0.02
            }
        }
        
    async def start_trading(self, asset: str, stop_event=None, mode: str = 'paper'):
        """Start trading for a specific asset."""
        if mode != 'paper':
            self.log.warning(f"Only paper trading is currently supported. Starting paper mode for {asset}")
        
        if asset not in self.paper_traders:
            self.log.error(f"Unknown asset type: {asset}")
            return
            
        if asset in self.trading_tasks and not self.trading_tasks[asset].done():
            self.log.warning(f"Trading already running for {asset}")
            return
            
        self.log.info(f"Starting paper trading for {asset}")
        
        # Update runtime state
        self.runtime.start_asset_trading(asset, 'paper')
        
        # Log to activity feed
        log_activity(
            source=asset.upper(),
            message=f"Paper trading started - Monitoring {', '.join(self.strategy_params[asset]['symbols'])}",
            type_="info"
        )
        
        # Start trading loop
        self.trading_tasks[asset] = asyncio.create_task(
            self._trading_loop(asset, stop_event)
        )
        
    async def stop_trading(self, asset: str):
        """Stop trading for a specific asset."""
        if asset in self.trading_tasks:
            self.log.info(f"Stopping trading for {asset}")
            self.trading_tasks[asset].cancel()
            try:
                await self.trading_tasks[asset]
            except asyncio.CancelledError:
                pass
            del self.trading_tasks[asset]
            
        # Update runtime state
        self.runtime.stop_asset_trading(asset, 'paper')
        
    async def _trading_loop(self, asset: str, stop_event=None):
        """Main trading loop for an asset."""
        trader = self.paper_traders[asset]
        params = self.strategy_params[asset]
        
        self.log.info(f"Trading loop started for {asset}")
        
        try:
            while True:
                # Check stop event if provided
                if stop_event and stop_event.is_set():
                    self.log.info(f"Trading loop stopped by event for {asset}")
                    break
                    
                # Check each symbol
                for symbol in params['symbols']:
                    await self._check_and_trade(asset, symbol, trader, params)
                
                # Wait before next check
                await asyncio.sleep(params['check_interval'])
                
        except asyncio.CancelledError:
            self.log.info(f"Trading loop cancelled for {asset}")
            raise
        except Exception as e:
            self.log.error(f"Error in trading loop for {asset}: {e}")
            raise
            
    async def _check_and_trade(self, asset: str, symbol: str, trader, params: Dict):
        """Check market and execute trades based on simple strategy."""
        try:
            # Get current market price from broker
            current_price = await self._get_market_price(asset, symbol)
            if not current_price:
                self.log.warning(f"Could not get price for {symbol}")
                return
            
            # Check if we have an open position
            open_position = None
            for pos in trader.positions:
                if pos['symbol'] == symbol and pos['status'] == 'open':
                    open_position = pos
                    break
                    
            if open_position:
                # Check if we should close the position
                entry_price = open_position['entry_price']
                pnl_pct = (current_price - entry_price) / entry_price
                
                if pnl_pct >= params['take_profit'] or pnl_pct <= params['stop_loss']:
                    # Close position
                    result = trader.close_position(
                        position_id=open_position['id'],
                        exit_price=current_price
                    )
                    
                    if result['success']:
                        self.log.info(
                            f"Closed {symbol} position for {asset}. "
                            f"P&L: ${result['pnl']:.2f} ({result['pnl_pct']:.2%})"
                        )
                        # Log to activity feed
                        log_activity(
                            source=asset.upper(),
                            message=f"Closed {symbol} position - P&L: ${result['pnl']:.2f} ({result['pnl_pct']:.2%})",
                            type_="trade"
                        )
                    
            else:
                # Generate signal using indicator-based strategy
                signal = self._generate_signal(asset, symbol, current_price)
                
                if signal != 'HOLD':
                    # Calculate position size
                    position_size = trader.balance * params['position_size']
                    
                    if position_size >= trader.min_trade_amount:
                        # Open position
                        result = trader.open_position(
                            symbol=symbol,
                            side=signal,
                            size_usd=position_size,
                            entry_price=current_price,
                            stop_loss_pct=params['stop_loss'],
                            take_profit_pct=params['take_profit']
                        )
                        
                        if result['success']:
                            self.log.info(
                                f"Opened {signal} position for {symbol} ({asset}). "
                                f"Size: ${position_size:.2f}, Price: ${current_price:.2f}"
                            )
                            # Log to activity feed
                            log_activity(
                                source=asset.upper(),
                                message=f"Opened {signal} position for {symbol} - Size: ${position_size:.2f}",
                                type_="trade"
                            )
                            
        except Exception as e:
            self.log.error(f"Error in check_and_trade for {asset}/{symbol}: {e}")
            
    def _initialize_brokers(self):
        """Initialize broker connections."""
        try:
            # Initialize Bybit for crypto
            self.brokers['crypto'] = ExchangeBybit("CRYPTO_SPOT", "paper")
            self.brokers['futures'] = ExchangeBybit("CRYPTO_FUTURES", "paper")
            self.log.info("Initialized Bybit connections for crypto trading")
            
            # Initialize IBKR for forex (if available)
            try:
                self.brokers['forex'] = IBKRConnectionManager()
                self.brokers['forex_options'] = self.brokers['forex']  # Same connection
                self.log.info("Initialized IBKR connection for forex trading")
            except Exception as e:
                self.log.warning(f"Could not initialize IBKR: {e}")
                self.brokers['forex'] = None
                self.brokers['forex_options'] = None
                
        except Exception as e:
            self.log.error(f"Error initializing brokers: {e}")
    
    async def _get_market_price(self, asset: str, symbol: str) -> Optional[float]:
        """Get real market price from broker. Retry on failure."""
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                broker = self.brokers.get(asset)
                if not broker:
                    self.log.error(f"No broker connection for {asset}")
                    await asyncio.sleep(retry_delay)
                    continue
                
                if asset in ['crypto', 'futures']:
                    # Fetch from Bybit
                    ticker = await broker.fetch_ticker(symbol)
                    if ticker and 'last' in ticker:
                        return float(ticker['last'])
                    
                elif asset in ['forex', 'forex_options'] and broker and broker.is_connected():
                    # Fetch from IBKR
                    contract = {'symbol': symbol[:3], 'currency': symbol[3:], 'secType': 'CASH'}
                    market_data = await broker.get_market_data(contract)
                    if market_data and 'last' in market_data:
                        return float(market_data['last'])
                else:
                    self.log.warning(f"Broker not connected for {asset}, retrying...")
                    await asyncio.sleep(retry_delay)
                    continue
                        
            except Exception as e:
                self.log.error(f"Error fetching market price for {symbol} (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
        
        self.log.error(f"Failed to get market price for {symbol} after {max_retries} attempts")
        return None
        
    def _initialize_strategies(self):
        """Initialize indicator-based strategies for each asset."""
        # Create strategy variants for each asset
        for asset in ['crypto', 'futures', 'forex', 'forex_options']:
            # Generate multiple strategy candidates
            variants = create_strategy_variants(asset, num_variants=5)
            
            # For now, pick the first variant for each asset
            # In full implementation, exploration manager would rotate these
            self.strategies[asset] = IndicatorStrategy(variants[0])
            
            self.log.info(f"Initialized {variants[0]['variant']} strategy for {asset}")
    
    def _generate_signal(self, asset: str, symbol: str, price: float) -> str:
        """Generate trading signal using indicator-based strategy."""
        if asset not in self.strategies:
            return 'HOLD'
        
        strategy = self.strategies[asset]
        signal, confidence = strategy.get_signal(symbol, price)
        
        # Only act on high confidence signals
        if confidence >= 0.5:
            self.log.info(f"Strategy {strategy.variant} for {asset} generated {signal} signal with {confidence:.2%} confidence")
            return signal
        
        return 'HOLD'
            
    def get_status(self) -> Dict[str, Any]:
        """Get current status of all trading activities."""
        status = {}
        
        for asset, trader in self.paper_traders.items():
            is_running = asset in self.trading_tasks and not self.trading_tasks[asset].done()
            
            status[asset] = {
                'running': is_running,
                'balance': trader.balance,
                'positions': len(trader.positions),
                'open_positions': len([p for p in trader.positions if p['status'] == 'open']),
                'total_trades': len(trader.trades),
                'pnl': trader.balance - trader.starting_balance,
                'pnl_pct': (trader.balance - trader.starting_balance) / trader.starting_balance * 100
            }
            
        return status


# Global trading engine instance
trading_engine = TradingEngine()
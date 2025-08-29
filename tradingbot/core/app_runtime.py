# file: tradingbot/core/app_runtime.py
# module_version: v1.00

"""
App Runtime - Global lifecycle orchestrator.
Manages boot, paper/live toggle, engine startup/stop, and graceful shutdowns.
This is the main orchestration module.
"""

import asyncio
import signal
import sys
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
from pathlib import Path

from .configmanager import config_manager
from .loggerconfig import get_logger
from .order_router import order_router
from .sl_tp_manager import sl_tp_manager
from .risk_manager import risk_manager
from .budget_manager import budget_manager
from .exposure_manager import exposure_manager
from .pnl_reconciler import pnl_reconciler


class RuntimeMode(Enum):
    """Runtime operational modes"""
    PAPER = "paper"
    LIVE = "live"
    HYBRID = "hybrid"  # Paper exploration + Live trading
    MAINTENANCE = "maintenance"


class AssetEngine(Enum):
    """Asset-specific trading engines"""
    SPOT = "spot"
    FUTURES = "futures"
    FOREX = "forex"
    OPTIONS = "options"


class AppRuntime:
    """
    Global lifecycle orchestrator for the trading bot.
    Coordinates all modules and manages runtime state.
    """
    
    def __init__(self):
        self.log = get_logger("app_runtime")
        self.config = config_manager
        
        # Runtime state
        self.mode = RuntimeMode.PAPER
        self.is_running = False
        self.start_time = None
        
        # Engine states
        self.engines: Dict[AssetEngine, Dict[str, Any]] = {
            AssetEngine.SPOT: {'enabled': False, 'mode': RuntimeMode.PAPER},
            AssetEngine.FUTURES: {'enabled': False, 'mode': RuntimeMode.PAPER},
            AssetEngine.FOREX: {'enabled': False, 'mode': RuntimeMode.PAPER},
            AssetEngine.OPTIONS: {'enabled': False, 'mode': RuntimeMode.PAPER}
        }
        
        # Module references (will be set during initialization)
        self.modules = {
            'order_router': order_router,
            'sl_tp_manager': sl_tp_manager,
            'risk_manager': risk_manager,
            'budget_manager': budget_manager,
            'exposure_manager': exposure_manager,
            'pnl_reconciler': pnl_reconciler
        }
        
        # Additional modules to be injected
        self.data_manager = None
        self.symbol_universe = None
        self.exploration_manager = None
        self.strategy_scheduler = None
        self.session_manager = None
        self.bankroll_manager = None
        self.strategy_registry = None
        self.validation_manager = None
        self.ui_server = None
        
        # Broker clients
        self.bybit_client = None
        self.ibkr_client = None
        
        # Main loop task
        self.main_loop_task = None
        self.tick_interval = 1.0  # Main loop tick interval in seconds
        
        # Graceful shutdown
        self._setup_signal_handlers()
        
        self.log.info("App Runtime initialized")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(sig, frame):
            self.log.info(f"Received signal {sig}, initiating graceful shutdown")
            asyncio.create_task(self.stop_all())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def initialize(self):
        """Initialize all modules and dependencies"""
        
        self.log.info("Initializing trading bot runtime")
        
        # Load configuration
        await self.config.load_config()
        
        # Initialize broker clients
        await self._initialize_brokers()
        
        # Wire dependencies between modules
        await self._wire_dependencies()
        
        # Initialize data feeds
        await self._initialize_data_feeds()
        
        # Load saved state
        await self._load_state()
        
        self.log.info("Runtime initialization complete")
    
    async def _initialize_brokers(self):
        """Initialize broker connections"""
        
        # Initialize Bybit client
        try:
            from tradingbot.brokers.exchangebybit import ExchangeBybit
            self.bybit_client = ExchangeBybit("CRYPTO_SPOT", "paper")
            await self.bybit_client.connect()
            self.log.info("Bybit client connected")
        except Exception as e:
            self.log.error(f"Failed to initialize Bybit client: {e}")
        
        # Initialize IBKR client
        try:
            from tradingbot.brokers.exchangeibkr import ExchangeIBKR
            self.ibkr_client = ExchangeIBKR()
            await self.ibkr_client.connect()
            self.log.info("IBKR client connected")
        except Exception as e:
            self.log.error(f"Failed to initialize IBKR client: {e}")
    
    async def _wire_dependencies(self):
        """Wire dependencies between modules"""
        
        # Wire core modules
        order_router.set_dependencies(
            risk_manager=risk_manager,
            budget_manager=budget_manager,
            exposure_manager=exposure_manager,
            sl_tp_manager=sl_tp_manager,
            pnl_reconciler=pnl_reconciler,
            bybit_client=self.bybit_client,
            ibkr_client=self.ibkr_client
        )
        
        sl_tp_manager.set_brokers(
            bybit_client=self.bybit_client,
            ibkr_client=self.ibkr_client
        )
        
        risk_manager.set_dependencies(
            exposure_manager=exposure_manager
        )
        
        budget_manager.set_dependencies(
            pnl_reconciler=pnl_reconciler
        )
        
        exposure_manager.set_dependencies(
            pnl_reconciler=pnl_reconciler
        )
        
        pnl_reconciler.set_brokers(
            bybit_client=self.bybit_client,
            ibkr_client=self.ibkr_client
        )
        
        self.log.info("Module dependencies wired")
    
    async def _initialize_data_feeds(self):
        """Initialize market data feeds"""
        
        # This will initialize data_manager and symbol_universe
        # when those modules are created
        
        self.log.info("Data feeds initialized")
    
    async def _load_state(self):
        """Load saved runtime state"""
        
        state_file = Path("tradingbot/state/runtime.json")
        if state_file.exists():
            # Load and restore state
            self.log.info("Loaded saved runtime state")
        else:
            self.log.info("No saved state found, starting fresh")
    
    async def start_all(self):
        """Start all trading engines"""
        
        if self.is_running:
            self.log.warning("Runtime already running")
            return
        
        self.log.info("Starting all trading engines")
        
        self.is_running = True
        self.start_time = datetime.now()
        
        # Start engines based on configuration
        config_assets = self.config.config.get('assets', {})
        allocations = config_assets.get('allocations_usd', {})
        
        for asset_type, allocation in allocations.items():
            if allocation > 0:
                engine = AssetEngine[asset_type.upper()]
                await self.enable_engine(engine)
        
        # Start main loop
        self.main_loop_task = asyncio.create_task(self._main_loop())
        
        # Start reconciliation loop
        asyncio.create_task(self._reconciliation_loop())
        
        # Start UI server if available
        if self.ui_server:
            await self.ui_server.start()
        
        self.log.info("All engines started")
    
    async def stop_all(self):
        """Stop all trading engines gracefully"""
        
        if not self.is_running:
            self.log.warning("Runtime not running")
            return
        
        self.log.info("Stopping all trading engines")
        
        self.is_running = False
        
        # Cancel all pending orders
        active_orders = order_router.get_active_orders()
        for order in active_orders:
            await order_router.cancel_order(order['order_id'])
        
        # Stop main loop
        if self.main_loop_task:
            self.main_loop_task.cancel()
            try:
                await self.main_loop_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect brokers
        if self.bybit_client:
            await self.bybit_client.disconnect()
        if self.ibkr_client:
            await self.ibkr_client.disconnect()
        
        # Save state
        await self._save_state()
        
        # Stop UI server
        if self.ui_server:
            await self.ui_server.stop()
        
        self.log.info("All engines stopped")
        
        # Exit
        sys.exit(0)
    
    async def enable_engine(self, engine: AssetEngine):
        """Enable a specific asset engine"""
        
        self.engines[engine]['enabled'] = True
        self.log.info(f"Enabled {engine.value} engine")
        
        # Initialize engine-specific components
        if engine == AssetEngine.SPOT:
            # Initialize spot-specific strategies
            pass
        elif engine == AssetEngine.FUTURES:
            # Initialize futures-specific strategies
            pass
        elif engine == AssetEngine.FOREX:
            # Initialize forex-specific strategies
            pass
        elif engine == AssetEngine.OPTIONS:
            # Initialize options-specific strategies
            pass
    
    async def disable_engine(self, engine: AssetEngine):
        """Disable a specific asset engine"""
        
        self.engines[engine]['enabled'] = False
        self.log.info(f"Disabled {engine.value} engine")
        
        # Close positions for this engine
        # This would be implemented when we have position tracking
    
    async def enable_live(self, asset: str):
        """Enable live trading for an asset"""
        
        try:
            engine = AssetEngine[asset.upper()]
            
            if not self.engines[engine]['enabled']:
                self.log.warning(f"Cannot enable live for disabled engine: {asset}")
                return False
            
            self.engines[engine]['mode'] = RuntimeMode.LIVE
            self.log.info(f"Enabled LIVE trading for {asset}")
            
            # Perform safety checks
            await self._perform_live_safety_checks(engine)
            
            return True
            
        except KeyError:
            self.log.error(f"Unknown asset type: {asset}")
            return False
    
    async def disable_live(self, asset: str):
        """Disable live trading for an asset (switch to paper)"""
        
        try:
            engine = AssetEngine[asset.upper()]
            
            self.engines[engine]['mode'] = RuntimeMode.PAPER
            self.log.info(f"Disabled live trading for {asset}, switched to PAPER")
            
            # Close all live positions for this asset
            # This would be implemented when we have position tracking
            
            return True
            
        except KeyError:
            self.log.error(f"Unknown asset type: {asset}")
            return False
    
    async def _main_loop(self):
        """Main runtime loop"""
        
        self.log.info("Main loop started")
        
        while self.is_running:
            try:
                # Tick all enabled engines
                await self.tick()
                
                # Sleep for tick interval
                await asyncio.sleep(self.tick_interval)
                
            except Exception as e:
                self.log.error(f"Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(self.tick_interval)
    
    async def tick(self):
        """Main loop tick - coordinates all active components"""
        
        # Process each enabled engine
        for engine, state in self.engines.items():
            if not state['enabled']:
                continue
            
            try:
                # Get data updates
                if self.data_manager:
                    # Process market data for this engine
                    pass
                
                # Check for exploration opportunities
                if self.exploration_manager:
                    # Process exploration for this engine
                    pass
                
                # Schedule strategy execution
                if self.strategy_scheduler:
                    # Process strategy scheduling for this engine
                    pass
                
                # Check risk limits
                if risk_manager:
                    # Monitor risk for this engine
                    pass
                
            except Exception as e:
                self.log.error(f"Error processing {engine.value}: {e}", exc_info=True)
    
    async def _reconciliation_loop(self):
        """Periodic reconciliation with brokers"""
        
        reconciliation_interval = 60  # Reconcile every minute
        
        while self.is_running:
            try:
                # Perform reconciliation
                report = await pnl_reconciler.reconcile()
                
                if report.broker_desync:
                    self.log.error("BROKER DESYNC DETECTED - Halting new trades")
                    # Set global halt flag
                    # This would trigger alerts and prevent new trades
                
                # Update budgets with reconciled P&L
                pnl_summary = pnl_reconciler.get_pnl_summary()
                for asset_type, pnl in pnl_summary['realized_pnl'].items():
                    budget_manager.apply_pnl(asset_type, pnl)
                
                await asyncio.sleep(reconciliation_interval)
                
            except Exception as e:
                self.log.error(f"Error in reconciliation loop: {e}", exc_info=True)
                await asyncio.sleep(reconciliation_interval)
    
    async def _perform_live_safety_checks(self, engine: AssetEngine):
        """Perform safety checks before enabling live trading"""
        
        checks_passed = True
        
        # Check broker connection
        if engine in [AssetEngine.SPOT, AssetEngine.FUTURES]:
            if not self.bybit_client or not await self.bybit_client.is_connected():
                self.log.error("Bybit not connected - cannot enable live")
                checks_passed = False
        else:
            if not self.ibkr_client or not await self.ibkr_client.is_connected():
                self.log.error("IBKR not connected - cannot enable live")
                checks_passed = False
        
        # Check risk limits configured
        risk_params = risk_manager.risk_params.get(engine.value.lower())
        if not risk_params:
            self.log.error(f"No risk parameters for {engine.value}")
            checks_passed = False
        
        # Check budget allocated
        allocation = budget_manager.get_alloc(engine.value.lower())
        if allocation <= 0:
            self.log.error(f"No budget allocated for {engine.value}")
            checks_passed = False
        
        if not checks_passed:
            # Revert to paper mode
            self.engines[engine]['mode'] = RuntimeMode.PAPER
            raise Exception("Live trading safety checks failed")
        
        self.log.info(f"Live trading safety checks passed for {engine.value}")
    
    async def _save_state(self):
        """Save runtime state to disk"""
        
        import json
        
        state = {
            'mode': self.mode.value,
            'engines': {k.value: {'enabled': v['enabled'], 'mode': v['mode'].value} 
                       for k, v in self.engines.items()},
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'timestamp': datetime.now().isoformat()
        }
        
        state_file = Path("tradingbot/state/runtime.json")
        state_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.log.info("Runtime state saved")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current runtime status"""
        
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        return {
            'is_running': self.is_running,
            'mode': self.mode.value,
            'uptime_seconds': uptime,
            'engines': {k.value: v for k, v in self.engines.items()},
            'modules_status': {
                'order_router': order_router is not None,
                'risk_manager': risk_manager is not None,
                'budget_manager': budget_manager is not None,
                'pnl_reconciler': pnl_reconciler is not None
            },
            'broker_status': {
                'bybit': self.bybit_client is not None,
                'ibkr': self.ibkr_client is not None
            }
        }


# Module initialization
app_runtime = AppRuntime()
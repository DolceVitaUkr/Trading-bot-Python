import argparse
import logging
import sys
import time
import json
from datetime import datetime, timezone
from typing import Optional

import config
from utils.utilities import configure_logging

# --- Module Imports ---
from modules.Funds_Controller import FundsController
from modules.Wallet_Sync import WalletSync
from modules.Portfolio_Manager import PortfolioManager
from modules.Sizer import Sizer
from modules.Strategy_Manager import StrategyManager
from modules.risk_management import RiskManager
from modules.trade_executor import TradeExecutor
from modules.reward_system import RewardSystem
from modules.data_manager import DataManager
from modules.exchange import ExchangeAPI
# Import new modules for integration
from modules.Validation_Manager import ValidationManager
from modules.News_Agent import NewsAgent
from modules.Kill_Switch import KillSwitch


# Placeholder for modules not in the scope of this refactoring
class MarketSessions:
    @staticmethod
    def current_session(now_utc: datetime) -> str:
        hour = now_utc.hour
        if 2 <= hour < 8: return "ASIA"
        if 7 <= hour < 16: return "EU"
        if 13 <= hour < 22: return "US"
        return "OVERLAP"

def run_bot(args: argparse.Namespace):
    """
    Initializes and runs the trading bot in a synchronous loop.
    """
    log = logging.getLogger()
    log.info("Booting modular trading bot...")

    try:
        with open("policies/sizing_policy.json", 'r') as f:
            sizing_policy = json.load(f)
    except Exception as e:
        log.exception(f"FATAL: Failed to load sizing_policy.json: {e}")
        return 1

    # --- Initialize Core Modules ---
    mock_exchange = ExchangeAPI() # For simulation, one exchange can serve all
    data_manager = DataManager(exchange=mock_exchange)

    exchange_adapters = {"SPOT": mock_exchange, "PERP": mock_exchange}

    funds_controller = FundsController()
    wallet_sync = WalletSync(exchange_adapters=exchange_adapters)

    allocations = getattr(config, "ASSET_ALLOCATION_USD", {"SPOT": 10000, "PERP": 5000})
    portfolio_manager = PortfolioManager(allocations=allocations, wallet_sync=wallet_sync)

    sizer = Sizer(policy=sizing_policy)
    # --- Initialize New Modules ---
    validation_config = {"min_trades_for_approval": 500}
    validation_manager = ValidationManager(config=validation_config)

    news_config = {
        "news_api_key": None, # No real API key for now
        "high_impact_events": ["CPI", "NFP", "FOMC", "ECB"]
    }
    news_agent = NewsAgent(config=news_config)

    kill_switch_config = {
        "daily_drawdown_limit": 0.05,
        "monthly_drawdown_limit": 0.15,
        "max_slippage_events": 3,
        "max_api_errors": 10
    }
    kill_switch = KillSwitch(config=kill_switch_config, portfolio_manager=portfolio_manager)

    # Use one ledger's balance for initializing RiskManager's equity tracking
    initial_rm_balance = sum(l['total'] for l in portfolio_manager.ledgers.values())
    risk_manager = RiskManager(
        account_balance=initial_rm_balance,
        sizing_policy=sizing_policy,
        kill_switch=kill_switch,
        data_provider=data_manager
    )

    strategy_manager = StrategyManager(
        data_provider=data_manager,
        validation_manager=validation_manager,
        news_agent=news_agent,
        portfolio_manager=portfolio_manager
    )
    trade_executor = TradeExecutor(sizing_policy=sizing_policy, simulation_mode=True, exchange=mock_exchange)
    reward_system = RewardSystem(starting_balance=initial_rm_balance)

    log.info("All modules initialized successfully.")

    # --- Main Trading Loop ---
    symbols_to_trade = ["BTC/USDT", "ETH/USDT"]
    asset_classes = {"BTC/USDT": "SPOT", "ETH/USDT": "SPOT"}

    while True:
        now_utc = datetime.now(timezone.utc)
        log.info(f"--- Starting new loop at {now_utc.isoformat()} ---")

        # --- Kill Switch Checks ---
        kill_switch.check_drawdowns()
        # Mock data for other checks for now
        kill_switch.check_slippage(slippage_events=[])
        kill_switch.check_api_errors(api_error_counts={})

        wallet_sync.sync()

        for symbol in symbols_to_trade:
            asset_class = asset_classes.get(symbol)
            if not asset_class: continue

            log.info(f"--- Evaluating {symbol} ({asset_class}) ---")

            if not funds_controller.is_allowed(asset_class, symbol):
                log.warning(f"Trading disabled for {asset_class} by FundsController. Skipping.")
                continue

            df_15m = data_manager.load_historical_data(symbol, "15m", backfill_bars=300)
            if df_15m.empty:
                log.warning(f"No 15m data for {symbol}. Skipping.")
                continue

            regime, regime_context = strategy_manager._determine_regime(df_15m)
            if regime == "Neutral":
                log.info(f"Neutral regime for {symbol}. No action.")
                continue

            session = MarketSessions.current_session(now_utc)
            mode = strategy_manager.select_mode(regime, session, asset_class)

            decision_context = {**regime_context, "regime": regime, "session": session, "asset_class": asset_class, "mode": mode}
            decision = strategy_manager.decide(symbol, decision_context)

            if not decision:
                log.info(f"No entry signal from strategy for {symbol}.")
                continue

            log.info(f"Signal found for {symbol}: {decision.signal}, Score: {decision.meta['signal_score']:.2f}")

            equity = portfolio_manager.available_budget(asset_class)
            price = df_15m['close'].iloc[-1]
            atr = regime_context.get('atr_15m', 0)

            proposal = sizer.propose(
                equity=equity, asset_class=asset_class, mode=mode, atr=atr, price=price,
                pair_cap_pct=funds_controller.pair_cap_pct(),
                signal_score=decision.meta['signal_score'], good_setup=decision.meta['good_setup']
            )

            if not proposal:
                log.warning(f"Sizer did not produce a valid proposal for {symbol}. Skipping.")
                continue

            log.info(f"Sizer proposed: SizeUSD={proposal['size_usd']:.2f}, Leverage={proposal['leverage']:.1f}x")

            is_allowed, reason = risk_manager.allow(
                proposal=proposal,
                asset_class=asset_class,
                symbol=symbol,
                side=decision.signal,
                price=price,
                mode=mode,
                session=session
            )
            if not is_allowed:
                log.warning(f"Trade rejected by RiskManager: {reason}. Skipping.")
                continue

            rid = portfolio_manager.reserve(asset_class, proposal['size_usd'])
            if not rid:
                log.error(f"Failed to reserve funds for {symbol}. Skipping.")
                continue

            receipt = trade_executor.execute_order(decision, proposal['size_usd'], proposal['leverage'], price)

            portfolio_manager.book_trade(asset=asset_class, pnl_net=receipt['pnl_net_usd'], fees=receipt['fees_usd'])
            reward_system.update(pnl_net_usd=receipt['pnl_net_usd'], context={"asset": asset_class, "mode": mode})
            portfolio_manager.release(rid)

            log.info(f"--- Completed trade cycle for {symbol} ---")

        loop_interval = getattr(config, "LIVE_LOOP_INTERVAL", 60)
        log.info(f"Loop finished. Sleeping for {loop_interval} seconds...")
        time.sleep(float(loop_interval))

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Modular Trading Bot")
    p.add_argument("--mode", choices=["simulation", "production"], default="simulation")
    return p.parse_args(argv)

if __name__ == "__main__":
    configure_logging()
    args = parse_args()
    try:
        sys.exit(run_bot(args))
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception:
        logging.getLogger().exception("Unhandled exception in main.")
        sys.exit(1)

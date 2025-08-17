"""
Main orchestrator for the trading bot.
"""
import os
import time
import datetime
import orjson
import asyncio
from dotenv import load_dotenv
from typing import Dict, Any, List

from core.interfaces import MarketData, Execution, WalletSync, NewsFeed, ValidationRunner
from core.schemas import DecisionTrace, StrategyMeta
from adapters.null_adapters import NullMarketData, NullExecution, NullWalletSync, NullNewsFeed
from adapters.news_rss import NewsRssAdapter
from adapters.ibkr_market import IbkrMarketData
from adapters.ibkr_exec import IbkrExecution
from adapters.wallet_bybit import BybitWalletSync
from adapters.wallet_ibkr import IbkrWalletSync
from adapters.composite_wallet import CompositeWalletSync
from managers.validation_manager import ValidationManager
from managers.sizer import Sizer
from managers.kill_switch import KillSwitch
from managers.strategy_manager import StrategyManager
from ib_insync import IB, util

# --- JSONL Logger ---
def append_to_jsonl(file_path: str, data: Dict[str, Any]):
    """Appends a JSON object to a JSONL file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "ab") as f:
        f.write(orjson.dumps(data, default=str))
        f.write(b"\n")

# --- Bot Factory ---
async def _build_bot():
    """
    Builds the bot with dependency injection based on environment variables.
    """
    load_dotenv()

    # --- Feature Flags ---
    enable_news = os.getenv("ENABLE_NEWS", "0") == "1"
    enable_forex = os.getenv("ENABLE_FOREX", "0") == "1"

    # --- Adapters ---
    market_data: MarketData = NullMarketData()
    execution: Execution = NullExecution()
    wallet_adapters: List[WalletSync] = []
    ib_client = None

    news_feed: NewsFeed = NewsRssAdapter() if enable_news else NullNewsFeed()
    print(f"News feature {'ENABLED' if enable_news else 'DISABLED'}.")

    if not enable_forex:
        print("Crypto mode enabled.")
        # Configure Bybit, etc.
    if enable_forex:
        print("Forex feature ENABLED. Connecting to IBKR...")
        ib_client = IB()
        # ... (IBKR connection logic as before) ...
        try:
            await ib_client.connectAsync(os.getenv("IBKR_HOST"), int(os.getenv("IBKR_PORT")), int(os.getenv("IBKR_CLIENT_ID")))
            market_data = IbkrMarketData(ib_client)
            execution = IbkrExecution(ib_client, market_data)
            wallet_adapters.append(IbkrWalletSync(ib_client))
            print("IBKR connection successful.")
        except Exception as e:
            print(f"FATAL: IBKR connection failed: {e}.")
            ib_client = None

    # --- Managers ---
    wallet_sync = CompositeWalletSync(wallet_adapters) if wallet_adapters else NullWalletSync()
    sizer = Sizer(wallet_sync)
    kill_switch = KillSwitch()
    validation_runner: ValidationRunner = ValidationManager()
    strategy_manager = StrategyManager()

    return {
        "market_data": market_data, "execution": execution, "wallet_sync": wallet_sync,
        "news_feed": news_feed, "validation_runner": validation_runner, "sizer": sizer,
        "kill_switch": kill_switch, "strategy_manager": strategy_manager,
        "ib_client": ib_client, "is_forex": enable_forex,
    }

# --- Main Loop ---
async def main():
    """ Main trading loop. """
    print("Building bot...")
    bot_components = await _build_bot()
    print("Bot built successfully.")

    # Unpack components
    market_data, execution, news_feed = bot_components["market_data"], bot_components["execution"], bot_components["news_feed"]
    validation_runner, sizer, kill_switch = bot_components["validation_runner"], bot_components["sizer"], bot_components["kill_switch"]
    strategy_manager = bot_components["strategy_manager"]
    ib_client, is_forex = bot_components["ib_client"], bot_components["is_forex"]

    # --- Strategy & Market Config ---
    if is_forex:
        strategy_id, symbol, market, asset_class, venue = "fx_strategy_01", "EUR/USD", "FX", "Forex", "IBKR"
        strategy_meta = StrategyMeta(strategy_id=strategy_id, name="SimpleFXMomentum", asset_class="Forex", market="FX", session_flags=["eu", "us"], timeframe="1h", indicators=["SMA"], params={"period": 50}, version="1.0", created_at=datetime.datetime.now(datetime.timezone.utc))
    else:
        strategy_id, symbol, market, asset_class, venue = "crypto_strategy_01", "BTC/USDT", "SPOT", "Crypto", "Bybit"
        strategy_meta = StrategyMeta(strategy_id=strategy_id, name="SimpleCryptoMomentum", asset_class="Crypto", market="SPOT", session_flags=["24/7"], timeframe="4h", indicators=["RSI"], params={"period": 14}, version="1.0", created_at=datetime.datetime.now(datetime.timezone.utc))

    # Register the strategy
    strategy_manager.register_strategy(strategy_meta)

    print(f"\n--- Starting main loop for {asset_class} on {venue} ---")
    try:
        # Dummy state for kill switch simulation
        simulated_pnl = 0

        while True:
            ts = datetime.datetime.now(datetime.timezone.utc)
            print(f"\n--- Iteration @ {ts.isoformat()} ---")

            blocked_reason, order_result, sizing_info = None, None, {}

            # 1. Kill Switch Check
            # In a real bot, pnl and position count would come from a portfolio manager
            simulated_pnl -= 1000 # Simulate a losing streak
            kill_switch.check_and_update(asset_class, venue, current_pnl=simulated_pnl, num_positions=5)

            is_blocked, block_rule = kill_switch.is_trading_blocked(asset_class, venue)
            if is_blocked:
                blocked_reason = f"KILL_SWITCH_ACTIVE:{block_rule}"

            # 2. Validation
            elif not (await validation_runner.approved(strategy_id, market))[0]:
                blocked_reason = "STRATEGY_NOT_APPROVED"

            # 3. Signal & Filters
            if not blocked_reason:
                signal = {"side": "buy", "score": 0.8}
                news_block = news_feed.macro_blockers([symbol]).get(symbol, False)
                filters = {"news_block": news_block}

                if news_block:
                    blocked_reason = "news_macro_block"
                else:
                    # 4. Sizing
                    price_data = await market_data.ticker(symbol)
                    price = price_data.get("price", 0.0)

                    if price > 0:
                        qty = await sizer.get_order_qty(asset_class, price)
                        sizing_info = {"qty": qty, "price": price}

                        if qty > 0:
                            # 5. Execution
                            order_result = await execution.place_order(symbol, side=signal["side"], qty=qty)
                            if order_result and order_result.get("status") == "REJECTED":
                                blocked_reason = order_result.get("reason")
                        else:
                            blocked_reason = "SIZING_QTY_ZERO"
                    else:
                        blocked_reason = "INVALID_PRICE_ZERO"

            # 6. Logging
            print(f"Blocked: {blocked_reason}" if blocked_reason else f"Order Result: {order_result}")
            trace = DecisionTrace(
                ts=ts, venue=venue, asset_class=asset_class, symbol=symbol, mode="paper",
                signal=signal if 'signal' in locals() else {}, filters=filters if 'filters' in locals() else {}, sizing=sizing_info,
                order_result=order_result, blocked_reason=blocked_reason,
                costs={"fee_est": 0.0, "slip_est": 0.0},
            )
            append_to_jsonl("state/decision_traces.jsonl", trace.dict(by_alias=True))
            print("Decision trace logged.")

            await asyncio.sleep(15)

    except KeyboardInterrupt:
        print("\n--- Shutting down ---")
    finally:
        if ib_client and ib_client.isConnected():
            print("Disconnecting from IBKR...")
            ib_client.disconnect()

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    util.logToConsole()
    asyncio.run(main())

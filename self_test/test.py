# Self_test/test.py

import os
import sys
import time
import traceback
from decimal import Decimal

# Ensure repo root on path when running directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RESULTS = []


def _ok(name, msg="ok"):
    RESULTS.append(("OK", name, msg))


def _fail(name, err):
    RESULTS.append(("FAIL", name, err))


def main():
    start = time.time()

    # 1) Imports & config presence
    try:
        import config  # noqa: F401
        _ok("config import")
    except Exception as e:
        _fail("config import", repr(e))

    # 2) Core modules import
    try:
        from modules.exchange import ExchangeAPI  # noqa
        from modules.trade_executor import TradeExecutor  # noqa
        from modules.technical_indicators import sma, rsi, atr  # noqa
        from modules.telegram_bot import TelegramNotifier  # noqa
        from modules.health_monitor import HealthMonitor  # noqa
        _ok("core modules import")
    except Exception:
        _fail("core modules import", traceback.format_exc())

    # 3) Technical indicators quick check
    try:
        series = [i for i in range(1, 101)]
        _ = sma(series, 20)
        _ = rsi(series, 14)
        highs = [i + 2 for i in series]
        lows = [i - 2 for i in series]
        closes = series[:]
        _ = atr(highs, lows, closes, 14)
        _ok("technical_indicators basic")
    except Exception:
        _fail("technical_indicators basic", traceback.format_exc())

    # 4) Exchange & TradeExecutor (simulation-friendly)
    try:
        from modules.trade_executor import TradeExecutor
        execu = TradeExecutor(simulation_mode=True)
        bal0 = execu.get_balance()
        # Use BTC/USDT by default; price fetch falls back to 0 only in extreme cases
        price = execu.exchange.get_price("BTC/USDT") or 50_000
        res_open = execu.execute_order("BTC/USDT", "buy", quantity=Decimal("0.0003"), price=Decimal(str(price)))
        assert res_open["status"] in ("open", "simulated")
        res_close = execu.close_position("BTC/USDT")
        bal1 = execu.get_balance()
        _ok("trade_executor sim", f"bal0={bal0:.2f} bal1={bal1:.2f}")
    except Exception:
        _fail("trade_executor sim", traceback.format_exc())

    # 5) Telegram (optional)
    try:
        token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        chat = os.getenv("TELEGRAM_CHAT_ID", "")
        if token and chat:
            from modules.telegram_bot import TelegramNotifier
            tn = TelegramNotifier(disable_async=True)
            tn.send_message_sync("🧪 Self-test: Telegram notifier looks good.", format="text")
            _ok("telegram send")
        else:
            _ok("telegram send", "skipped (no token/chat env)")
    except Exception:
        _fail("telegram send", traceback.format_exc())

    # 6) Health monitor quick spin
    try:
        from modules.health_monitor import HealthMonitor
        from modules.trade_executor import TradeExecutor

        execu = TradeExecutor(simulation_mode=True)

        def status_provider():
            # Best-effort status dict
            return {
                "balance": execu.get_balance(),
                "equity": execu.get_balance(),  # unrealized not tracked here
                "open_positions": len(execu.exchange.fetch_positions() if hasattr(execu.exchange, "fetch_positions") else []),
                "symbol": "BTC/USDT",
                "last_trade": "n/a",
            }

        hm = HealthMonitor(
            notifier=TelegramNotifier(disable_async=True),
            mode="paper",
            heartbeat_interval_sec=2,
            recap_interval_sec=3,
            watchdog_timeout_sec=6,
            quiet_heartbeat=True,
        )
        hm.set_status_provider(status_provider)
        hm.start()
        # Feed a few heartbeats
        for _ in range(3):
            hm.record_heartbeat("self_test")
            time.sleep(1.0)
        # allow one recap to attempt (will log to stdout if no telegram creds)
        time.sleep(4.0)
        hm.stop()
        _ok("health_monitor basic")
    except Exception:
        _fail("health_monitor basic", traceback.format_exc())

    # 7) Write summary
    dur = time.time() - start
    lines = []
    ok_count = sum(1 for s, *_ in RESULTS if s == "OK")
    fail_count = sum(1 for s, *_ in RESULTS if s == "FAIL")
    lines.append("==== Self Test Report ====")
    lines.append(f"Duration: {dur:.2f}s")
    lines.append(f"OK: {ok_count} | FAIL: {fail_count}")
    for status, name, info in RESULTS:
        lines.append(f"[{status}] {name} - {info}")

    report_path = os.path.join(os.path.dirname(__file__), "report.txt")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception:
        pass

    print("\n".join(lines))
    print(f"(report saved to {report_path})")


if __name__ == "__main__":
    import os
    import sys
    main()

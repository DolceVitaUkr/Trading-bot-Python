# state/position_reconciler.py

import asyncio
import logging
from typing import List, Dict, Any, Optional

from modules.exchange import ExchangeAPI  # uses reconcile_open_state()
from state.runtime_state import RuntimeState

try:
    from modules.telegram_bot import TelegramNotifier
except Exception:
    TelegramNotifier = None  # type: ignore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def _reconcile_one_async(
    ex: ExchangeAPI,
) -> Dict[str, Any]:
    """
    Async wrapper around ExchangeAPI.reconcile_open_state() (which is sync).
    """
    return await asyncio.to_thread(ex.reconcile_open_state)


async def reconcile_all_async(
    exchanges: List[ExchangeAPI],
    *,
    state: Optional[RuntimeState] = None,
    notifier: Optional["TelegramNotifier"] = None
) -> Dict[str, Any]:
    """
    Reconcile open positions/orders across all provided exchanges.

    - Calls each exchange's reconcile method.
    - Optionally updates RuntimeState open_positions.
    - Optionally sends a compact Telegram summary.

    Returns a merged summary:
    {
      "exchanges": [
        {"idx": 0, "summary": {...}},
        ...
      ],
      "actions": int_total_actions
    }
    """
    merged: Dict[str, Any] = {"exchanges": [], "actions": 0}

    async def _wrap(idx: int, ex: ExchangeAPI):
        try:
            s = await _reconcile_one_async(ex)
            merged["exchanges"].append({"idx": idx, "summary": s})
            merged["actions"] += len(s.get("actions", []))
            return s
        except Exception as e:
            logger.exception("reconcile failed for exchange idx=%s: %s", idx, e)
            merged["exchanges"].append({"idx": idx, "error": str(e)})
            return {}

    await asyncio.gather(*(_wrap(i, ex) for i, ex in enumerate(exchanges)))

    # Update state with any detected positions (best-effort)
    if state is not None:
        try:
            for item in merged["exchanges"]:
                summary = item.get("summary") or {}
                positions = summary.get("positions", [])
                # we don't know domain from exchange; assume "crypto"
                for pos in positions:
                    sym = pos.get("symbol")
                    if sym:
                        state.upsert_open_position("crypto", sym, pos)
            state.save()
        except Exception:
            logger.exception("Failed to persist reconciled positions to state")

    # Optional notifier summary
    if notifier is not None:
        try:
            total_pos = sum(len((x.get("summary") or {}).get("positions", [])) for x in merged["exchanges"])
            total_ord = sum(len((x.get("summary") or {}).get("orders", [])) for x in merged["exchanges"])
            total_act = merged.get("actions", 0)
            txt = (
                f"🔄 <b>Reconciliation Summary</b>\n"
                f"Exchanges: {len(exchanges)}\n"
                f"Open positions: {total_pos}\n"
                f"Open orders: {total_ord}\n"
                f"Protective actions taken: {total_act}"
            )
            notifier.send_message_sync(txt, format="text")
        except Exception:
            logger.exception("Failed to send reconciliation summary to Telegram")

    return merged

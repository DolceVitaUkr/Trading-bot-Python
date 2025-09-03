from typing import List, Dict, Any, Optional
from .loggerconfig import get_logger

log = get_logger(__name__)

class BybitCatalogFetcher:
    def __init__(self, adapter):
        self.adapter = adapter
    async def fetch_symbols(self, symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Return a list of contract dicts with tick/step/min_notional/multiplier/etc.
        This is a placeholder: implement adapter discovery calls here.
        """
        out: List[Dict[str, Any]] = []
        # Example structure per symbol
        sel = symbols or []
        for sym in sel:
            out.append({
                "id": sym,
                "type": "perp" if "PERP" in sym or "USDT" in sym else "future",
                "underlying": sym,
                "expiry": None,
                "multiplier": 1.0,
                "tick_size": 0.01,
                "lot_size": 0.001,
                "min_notional": 10.0,
                "settlement": "USDT",
                "exercise_style": None,
                "funding_schedule": "8h",
                "session_calendar": "BYBIT-PERP"
            })
        return out

class IBKRCatalogFetcher:
    def __init__(self, adapter):
        self.adapter = adapter
    async def fetch_symbols(self, symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Return a list of contract dicts using IBKR ContractDetails via adapter.
        This is a placeholder: implement ib_insync ContractDetails calls here.
        """
        out: List[Dict[str, Any]] = []
        for sym in (symbols or []):
            out.append({
                "id": sym,
                "type": "forex" if "." in sym else "option" if "OPT" in sym else "future",
                "underlying": sym.split(" ")[0] if " " in sym else sym,
                "expiry": None,
                "multiplier": 100.0 if "option" else 1.0,
                "tick_size": 0.0001,
                "lot_size": 1.0,
                "min_notional": 0.0,
                "settlement": "USD",
                "exercise_style": "American",
                "session_calendar": "IBKR-FX-OPT"
            })
        return out

async def refresh_contracts(catalog, bybit_adapter=None, ibkr_adapter=None, symbols: Optional[List[str]] = None):
    """Populate/refresh catalog using venue fetchers. If symbols is None, use your universe selection.

    Merge duplicates by id (IBKR overrides Bybit on conflicts).
    """
    items: Dict[str, Dict[str, Any]] = {}
    if bybit_adapter:
        bb = BybitCatalogFetcher(bybit_adapter)
        for c in await bb.fetch_symbols(symbols):
            items[c["id"]] = c
    if ibkr_adapter:
        ib = IBKRCatalogFetcher(ibkr_adapter)
        for c in await ib.fetch_symbols(symbols):
            items[c["id"]] = c
    final = list(items.values())
    catalog.refresh(final)
    log.info(f"Catalog refreshed with {len(final)} contracts")
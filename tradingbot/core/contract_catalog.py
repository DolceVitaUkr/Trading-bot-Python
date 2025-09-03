import json, time
from pathlib import Path
from typing import Dict, Any, Optional, List
from .loggerconfig import get_logger
log = get_logger(__name__)
DEFAULT_CATALOG = {"schema_version": 1, "generated_at": None, "contracts": []}
class ContractCatalog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.data = DEFAULT_CATALOG.copy()
    def load(self) -> None:
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = DEFAULT_CATALOG.copy()
            self.data["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self.save()
    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
    def refresh(self, contracts: Optional[List[Dict[str, Any]]] = None) -> None:
        if contracts is not None:
            self.data["contracts"] = contracts
        self.data["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.save()

    # NEW: convenience to refresh from venue adapters
    async def refresh_from_venues(self, bybit_adapter=None, ibkr_adapter=None, symbols: Optional[List[str]] = None) -> None:
        from .catalog_fetchers import refresh_contracts
        await refresh_contracts(self, bybit_adapter, ibkr_adapter, symbols)
    def find(self, contract_id: str) -> Optional[Dict[str, Any]]:
        for c in self.data.get("contracts", []):
            if c.get("id") == contract_id:
                return c
        return None
    def metadata_for_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        return self.find(symbol)
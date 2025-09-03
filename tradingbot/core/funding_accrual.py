import time
from typing import Dict, Any, List
from pathlib import Path
from .loggerconfig import get_logger
from .order_events import emit_event

log = get_logger(__name__)

class FundingAccrual:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
    def accrue(self, asset: str, positions: List[Dict[str, Any]], funding_rate: float):
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        for pos in positions:
            notional = float(pos.get('qty', 0)) * float(pos.get('mark_price', pos.get('avg_price', 0)))
            fee = notional * funding_rate
            emit_event(self.base_dir, 'paper', asset, {
                "event": "FUNDING_ACCRUAL", "contract_id": pos.get('contract_id'),
                "rate": funding_rate, "amount": fee
            })
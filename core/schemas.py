"""
Pydantic schemas for data validation and serialization.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import datetime

class StrategyMeta(BaseModel):
    """Metadata for a trading strategy."""
    strategy_id: str
    name: str
    asset_class: str = Field(..., alias="class") # 'class' is a reserved keyword
    market: str
    session_flags: List[str]
    timeframe: str
    indicators: List[str]
    params: Dict[str, Any]
    version: str
    created_at: datetime.datetime

class ValidationRecord(BaseModel):
    """Record of a strategy validation run."""
    strategy_id: str
    market: str
    period: str
    n_trades: int
    sharpe: float
    max_dd: float
    winrate: float
    avg_pnl: float
    slippage_mean: float
    decision: Dict[str, Any]
    promoted_at: Optional[datetime.datetime] = None
    cool_off_until: Optional[datetime.datetime] = None

class DecisionTrace(BaseModel):
    """Trace of a single trading decision."""
    ts: datetime.datetime
    venue: str
    asset_class: str
    symbol: str
    mode: str  # paper/live
    signal: Dict[str, Any]
    filters: Dict[str, Any]
    sizing: Dict[str, Any]
    order_result: Optional[Dict[str, Any]] = None
    blocked_reason: Optional[str] = None
    costs: Dict[str, Any]

class KillEvent(BaseModel):
    """Record of a kill switch event."""
    ts: datetime.datetime
    scope: Dict[str, str]
    rule: str
    action: Dict[str, Any]
    auto_rearm_at: datetime.datetime
    note: str

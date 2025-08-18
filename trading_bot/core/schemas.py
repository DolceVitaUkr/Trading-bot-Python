"""
Pydantic schemas for data validation and serialization.
"""
from typing import List, Dict, Any, Optional, Tuple, Literal
from pydantic import BaseModel, Field
import datetime
import uuid
from enum import Enum


class BranchStatus(str, Enum):
    """Enumeration for the status of a trading branch."""
    STOPPED = "stopped"
    RUNNING = "running"
    ERROR = "error"


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
    product: str # e.g., FOREX_SPOT, CRYPTO_SPOT
    market: Optional[str] = None # Market is now optional, product is primary
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

# -----------------------------------------------------------------------------
# Core Data Contracts
# -----------------------------------------------------------------------------

class OHLCV(BaseModel):
    """Represents a single OHLCV candle."""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

class MarketData(BaseModel):
    """Standardized market data contract."""
    symbol: str
    ohlcv: List[OHLCV] = []
    # Optional fields for other data types
    order_book: Optional[Dict[str, List[Tuple[float, float]]]] = None
    last_trade: Optional[Tuple[float, float]] = None # (price, size)

class OrderStatus(str, Enum):
    """Standardized order status enumeration."""
    OPEN = "open"
    CLOSED = "closed"
    CANCELED = "canceled"
    REJECTED = "rejected"
    PENDING = "pending"

class Order(BaseModel):
    """Standardized order contract for all modules."""
    order_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    order_type: Literal["market", "limit"] = "market"
    limit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    updated_at: Optional[datetime.datetime] = None

class Position(BaseModel):
    """Standardized position contract."""
    symbol: str
    side: Literal["long", "short"]
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    leverage: float
    liquidation_price: Optional[float] = None

class PortfolioState(BaseModel):
    """Standardized portfolio state contract."""
    total_balance_usd: float
    available_balance_usd: float
    margin_used: float
    unrealized_pnl: float
    realized_pnl: float
    positions: List[Position]

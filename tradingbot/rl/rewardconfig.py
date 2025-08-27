"""
Reward Configuration for Trading Bot RL System
Provides typed configuration with environment variable overrides.
"""
import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class RewardConfig:
    """Configuration for reward and penalty calculations."""
    
    # Performance targets
    targetwinrate: float = float(os.getenv('TARGET_WIN_RATE', 0.75))
    targetsharpe: float = float(os.getenv('TARGET_SHARPE', 2.0))
    targetavgprofitpertrade: float = float(os.getenv('TARGET_AVG_PROFIT', 0.30))
    
    # Risk limits
    maxstoplossfrac: float = float(os.getenv('MAX_STOP_LOSS_FRAC', 0.10))
    maxdrawdownfracsoft: float = float(os.getenv('MAX_DD_SOFT', 0.15))
    maxdrawdownfrachard: float = float(os.getenv('MAX_DD_HARD', 0.20))
    killswitchdrawdownfrac: float = float(os.getenv('KILL_SWITCH_DD', 0.30))
    
    # Stop loss penalties
    consecutivestoplossthreshold: int = int(os.getenv('CONSEC_SL_THRESHOLD', 5))
    consecutivestoplosspenalty: float = float(os.getenv('CONSEC_SL_PENALTY', -5.0))
    
    # Basic scoring
    losspointsper1perc: float = float(os.getenv('LOSS_POINTS_PER_PERC', -5.0))
    tradebonuspointsifprofitover10perc: float = float(os.getenv('TRADE_BONUS_10PERC', 3.0))
    
    # Profit bands: (min_profit_frac, points_awarded)
    profitbands: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.11, 1), (0.20, 5), (0.31, 10), (0.51, 20),
        (0.61, 70), (0.81, 200), (1.00, 500)
    ])
    
    # Shaping weights
    sharpebonusweight: float = float(os.getenv('SHARPE_BONUS_WEIGHT', 0.5))
    exposurepenaltyweight: float = float(os.getenv('EXPOSURE_PENALTY_WEIGHT', 1.0))
    leveragepenaltyweight: float = float(os.getenv('LEVERAGE_PENALTY_WEIGHT', 1.0))
    holdingtimepenaltyweight: float = float(os.getenv('HOLDING_TIME_PENALTY', 0.2))
    feesslippagepenaltyweight: float = float(os.getenv('FEES_PENALTY_WEIGHT', 1.0))
    drawdownpenaltyweight: float = float(os.getenv('DD_PENALTY_WEIGHT', 2.0))
    
    # Reward clipping
    clipstepreward: float = float(os.getenv('CLIP_STEP_REWARD', 1000.0))
    clipepisodereward: float = float(os.getenv('CLIP_EPISODE_REWARD', 5000.0))
    
    # EMA smoothing
    emaalpha: float = float(os.getenv('EMA_ALPHA', 0.1))
    
    # Risk management
    maxconcurrentexposurefrac: float = float(os.getenv('MAX_EXPOSURE_FRAC', 0.40))
    
    # Leverage tiers: (equity_threshold, max_leverage)
    leveragetiers: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0, 1), (1000, 3), (10000, 5), (50000, 10)
    ])
    
    # Stop loss violation penalty
    slviolationpenalty: float = float(os.getenv('SL_VIOLATION_PENALTY', -10.0))
    
    # File paths
    patheventsjsonl: str = os.getenv(
        'REWARD_EVENTS_PATH',
        'C:/Users/Asus/Documents/MASTERPIECE/backupdata/rlevents/events.jsonl'
    )
    pathtrademetricscsv: str = os.getenv(
        'TRADE_METRICS_PATH', 
        'C:/Users/Asus/Documents/MASTERPIECE/backupdata/rlmetrics/trademetrics.csv'
    )
    pathepisodemetricscsv: str = os.getenv(
        'EPISODE_METRICS_PATH',
        'C:/Users/Asus/Documents/MASTERPIECE/backupdata/rlmetrics/episodemetrics.csv'
    )
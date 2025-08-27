"""
Core Reward Engine for Trading Bot RL System
Provides deterministic reward calculations for reinforcement learning.
"""
import math
import statistics
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

from .rewardconfig import RewardConfig


@dataclass
class StepContext:
    """Context information for a single trading step."""
    symbol: str
    side: Optional[Literal["long", "short"]]
    entryprice: Optional[float]
    exitprice: Optional[float]
    midprice: Optional[float]
    qty: Optional[float]
    leverage: float
    feespaidusd: float
    slippageusd: float
    stoplossprice: Optional[float]
    takeprofitprice: Optional[float]
    holdingtimeseconds: float
    tradeclosed: bool
    realizedprofitperc: Optional[float]  # net of fees if tradeclosed
    equityusd: float
    peakequityusd: float
    openexposureusd: float
    consecutivestoplosshits: int
    rollingreturnswindow: List[float]  # last N step returns for Sharpe proxy
    dtseconds: float


@dataclass
class EpisodeSummary:
    """Summary metrics for a completed trading episode."""
    totaltrades: int
    wins: int
    losses: int
    grossreturnfrac: float
    maxdrawdownfrac: float
    sharpeestimate: Optional[float]
    equitypath: List[float]
    bandedprofitpoints: float
    bandedlosspoints: float


def drawdownfrac(equity: float, peak: float) -> float:
    """Calculate drawdown fraction from peak equity."""
    if peak <= 0:
        return 0.0
    return max(0.0, (peak - equity) / peak)


def leveragetiercap(cfg: RewardConfig, equity: float) -> float:
    """Determine maximum allowed leverage based on equity tier."""
    for threshold, max_lev in sorted(cfg.leveragetiers, reverse=True):
        if equity >= threshold:
            return max_lev
    return 1.0  # Default minimum leverage


def sharpeproxy(returns: List[float]) -> Optional[float]:
    """Estimate Sharpe ratio from returns window."""
    if len(returns) < 2:
        return None
    
    try:
        mean_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)
        
        if std_return == 0:
            return None
        
        return mean_return / std_return
    except (statistics.StatisticsError, ZeroDivisionError):
        return None


def normalizeandclip(value: float, clip_limit: float) -> float:
    """Normalize and clip reward value to prevent extreme values."""
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(-clip_limit, min(clip_limit, value))


def ema(new_value: float, old_ema: float, alpha: float) -> float:
    """Exponential moving average update."""
    return alpha * new_value + (1 - alpha) * old_ema


def isstoplossviolation(realized_loss_frac: float, max_sl_frac: float) -> bool:
    """Check if realized loss exceeds stop loss threshold."""
    return realized_loss_frac > max_sl_frac


def profitpointsfrombands(profit_frac: float, bands: List[Tuple[float, float]]) -> float:
    """Calculate points from profit bands. Bands must be sorted by profit threshold."""
    if profit_frac <= 0:
        return 0.0
    
    points = 0.0
    for min_profit, band_points in sorted(bands):
        if profit_frac >= min_profit:
            points = band_points
        else:
            break
    
    return points


def computestepreward(cfg: RewardConfig, s: StepContext) -> Tuple[float, Dict[str, any]]:
    """
    Compute reward for a single step with detailed component breakdown.
    
    Returns:
        Tuple of (reward_value, components_dict)
    """
    components = {
        "pnlpoints": 0.0,
        "losspoints": 0.0,
        "bandbonus": 0.0,
        "feespenalty": 0.0,
        "slviolation": 0.0,
        "exposurepenalty": 0.0,
        "leveragepenalty": 0.0,
        "holdingdecay": 0.0,
        "sharpebonus": 0.0,
        "ddsoftpenalty": 0.0,
        "consecutiveslpenalty": 0.0,
        "killswitch": False,
        "killreason": None,
        "error": None
    }
    
    reward = 0.0
    
    try:
        # 1. Check kill switch conditions
        dd = drawdownfrac(s.equityusd, s.peakequityusd)
        if dd >= cfg.killswitchdrawdownfrac:
            components["killswitch"] = True
            components["killreason"] = f"Drawdown {dd:.3f} >= {cfg.killswitchdrawdownfrac}"
            reward -= 500.0  # Hard penalty
        
        # 2. Process closed trade rewards/penalties
        if s.tradeclosed:
            if s.realizedprofitperc is None:
                components["error"] = "missingrealized"
                reward -= 100.0  # Hard penalty for missing data
            else:
                profit = s.realizedprofitperc
                
                # Profit band points
                if profit >= 0:
                    components["pnlpoints"] = profitpointsfrombands(profit, cfg.profitbands)
                    reward += components["pnlpoints"]
                else:
                    # Loss points (linear penalty) - for losses, we want negative points
                    # profit is negative (e.g., -0.08), losspointsper1perc is negative (e.g., -5.0)
                    # So: (-0.08 * 100) * (-5.0) = -8 * -5 = 40, but we want this to be negative
                    # Solution: use abs(profit) to make the calculation work correctly
                    components["losspoints"] = abs(profit * 100) * cfg.losspointsper1perc  # This gives negative points
                    reward += components["losspoints"]
                
                # Trade bonus for profitable trades
                if profit >= 0.10:
                    components["bandbonus"] = cfg.tradebonuspointsifprofitover10perc
                    reward += components["bandbonus"]
                
                # Fees and slippage penalty
                components["feespenalty"] = cfg.feesslippagepenaltyweight * (s.feespaidusd + s.slippageusd)
                reward -= components["feespenalty"]
                
                # Stop loss violation penalty
                if profit < 0 and isstoplossviolation(abs(profit), cfg.maxstoplossfrac):
                    components["slviolation"] = cfg.slviolationpenalty
                    reward += components["slviolation"]
        
        # 3. Continuous shaping penalties
        
        # Exposure penalty
        max_exposure = cfg.maxconcurrentexposurefrac * s.equityusd
        if s.openexposureusd > max_exposure:
            excess_exposure = s.openexposureusd - max_exposure
            components["exposurepenalty"] = cfg.exposurepenaltyweight * (excess_exposure / s.equityusd)
            reward -= components["exposurepenalty"]
        
        # Leverage penalty
        max_leverage = leveragetiercap(cfg, s.equityusd)
        if s.leverage > max_leverage:
            excess_leverage = s.leverage - max_leverage
            components["leveragepenalty"] = cfg.leveragepenaltyweight * excess_leverage
            reward -= components["leveragepenalty"]
        
        # Holding time decay
        holding_hours = s.holdingtimeseconds / 3600.0
        components["holdingdecay"] = cfg.holdingtimepenaltyweight * holding_hours
        reward -= components["holdingdecay"]
        
        # Consecutive stop loss penalty
        if s.consecutivestoplosshits >= cfg.consecutivestoplossthreshold:
            excess_sl = s.consecutivestoplosshits - cfg.consecutivestoplossthreshold
            components["consecutiveslpenalty"] = cfg.consecutivestoplosspenalty * excess_sl
            reward += components["consecutiveslpenalty"]
        
        # 4. Risk-adjusted shaping
        
        # Sharpe bonus
        sharpe = sharpeproxy(s.rollingreturnswindow)
        if sharpe is not None and sharpe > 1.0:
            components["sharpebonus"] = cfg.sharpebonusweight * (sharpe - 1.0)
            reward += components["sharpebonus"]
        
        # Soft drawdown penalty
        if dd > cfg.maxdrawdownfracsoft:
            excess_dd = dd - cfg.maxdrawdownfracsoft
            components["ddsoftpenalty"] = cfg.drawdownpenaltyweight * excess_dd
            reward -= components["ddsoftpenalty"]
        
        # 5. Final clipping
        reward = normalizeandclip(reward, cfg.clipstepreward)
        
    except Exception as e:
        components["error"] = str(e)
        reward = -50.0  # Penalty for calculation errors
    
    return reward, components


def computeepisodereward(cfg: RewardConfig, ep: EpisodeSummary) -> Tuple[float, Dict[str, any]]:
    """
    Compute reward for a completed episode.
    
    Returns:
        Tuple of (reward_value, components_dict)
    """
    components = {
        "bandedpoints": 0.0,
        "winratebonus": 0.0,
        "sharpebonus": 0.0,
        "avgprofitbonus": 0.0,
        "ddpenalty": 0.0,
        "error": None
    }
    
    reward = 0.0
    
    try:
        # Base points from banded profits and losses
        components["bandedpoints"] = ep.bandedprofitpoints + ep.bandedlosspoints
        reward += components["bandedpoints"]
        
        # Performance bonuses if targets are met
        if ep.totaltrades > 0:
            win_rate = ep.wins / ep.totaltrades
            if win_rate >= cfg.targetwinrate:
                components["winratebonus"] = 50.0 * (win_rate - cfg.targetwinrate)
                reward += components["winratebonus"]
            
            avg_profit_per_trade = ep.grossreturnfrac / ep.totaltrades
            if avg_profit_per_trade >= cfg.targetavgprofitpertrade:
                components["avgprofitbonus"] = 30.0 * (avg_profit_per_trade - cfg.targetavgprofitpertrade)
                reward += components["avgprofitbonus"]
        
        # Sharpe bonus
        if ep.sharpeestimate is not None and ep.sharpeestimate >= cfg.targetsharpe:
            components["sharpebonus"] = 100.0 * (ep.sharpeestimate - cfg.targetsharpe)
            reward += components["sharpebonus"]
        
        # Drawdown penalties
        if ep.maxdrawdownfrac > cfg.maxdrawdownfrachard:
            components["ddpenalty"] = -200.0 * (ep.maxdrawdownfrac - cfg.maxdrawdownfrachard)
            reward += components["ddpenalty"]
        elif ep.maxdrawdownfrac > cfg.maxdrawdownfracsoft:
            components["ddpenalty"] = -50.0 * (ep.maxdrawdownfrac - cfg.maxdrawdownfracsoft)
            reward += components["ddpenalty"]
        
        # Final clipping
        reward = normalizeandclip(reward, cfg.clipepisodereward)
        
    except Exception as e:
        components["error"] = str(e)
        reward = -100.0  # Penalty for calculation errors
    
    return reward, components
"""
Reward Integration Adapters for Trading Bot RL System
Provides safe integration with existing training loop and logging systems.
"""
import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .rewardconfig import RewardConfig
from .rewardengine import (
    StepContext, 
    EpisodeSummary, 
    computestepreward, 
    computeepisodereward
)


class RewardAdapters:
    """Handles logging and notification adapters for reward system."""
    
    def __init__(
        self, 
        logger: logging.Logger, 
        sendtelegramfn: Optional[Callable[[str], None]] = None, 
        cfg: RewardConfig = None
    ):
        self.logger = logger
        self.sendtelegramfn = sendtelegramfn
        self.cfg = cfg or RewardConfig()
    
    def log_step_reward(
        self, 
        ctx: StepContext, 
        reward: float, 
        components: Dict[str, Any]
    ) -> None:
        """Log detailed step reward information."""
        try:
            # Create human-readable summary
            summary_parts = []
            
            if ctx.tradeclosed:
                profit_pct = ctx.realizedprofitperc or 0.0
                summary_parts.append(f"Trade closed: {ctx.symbol} {profit_pct:.2%}")
                
                if components.get("pnlpoints", 0) > 0:
                    summary_parts.append(f"Profit points: +{components['pnlpoints']:.1f}")
                if components.get("losspoints", 0) < 0:
                    summary_parts.append(f"Loss points: {components['losspoints']:.1f}")
                if components.get("bandbonus", 0) > 0:
                    summary_parts.append(f"Band bonus: +{components['bandbonus']:.1f}")
            
            penalties = []
            if components.get("feespenalty", 0) > 0:
                penalties.append(f"fees: -{components['feespenalty']:.1f}")
            if components.get("exposurepenalty", 0) > 0:
                penalties.append(f"exposure: -{components['exposurepenalty']:.1f}")
            if components.get("leveragepenalty", 0) > 0:
                penalties.append(f"leverage: -{components['leveragepenalty']:.1f}")
            if components.get("holdingdecay", 0) > 0:
                penalties.append(f"holding: -{components['holdingdecay']:.1f}")
            if components.get("ddsoftpenalty", 0) > 0:
                penalties.append(f"drawdown: -{components['ddsoftpenalty']:.1f}")
            
            if penalties:
                summary_parts.append(f"Penalties: {', '.join(penalties)}")
            
            bonuses = []
            if components.get("sharpebonus", 0) > 0:
                bonuses.append(f"sharpe: +{components['sharpebonus']:.1f}")
            
            if bonuses:
                summary_parts.append(f"Bonuses: {', '.join(bonuses)}")
            
            summary = " | ".join(summary_parts) if summary_parts else "No significant activity"
            
            self.logger.info(f"Step reward: {reward:.2f} | {summary}")
            
            # Kill switch warning
            if components.get("killswitch", False):
                kill_msg = f"🚨 KILL SWITCH ACTIVATED: {components.get('killreason', 'Unknown')}"
                self.logger.error(kill_msg)
                if self.sendtelegramfn:
                    self.sendtelegramfn(kill_msg)
            
            # Error logging
            if components.get("error"):
                error_msg = f"Reward calculation error: {components['error']}"
                self.logger.error(error_msg)
            
        except Exception as e:
            self.logger.error(f"Error logging step reward: {e}")
    
    def log_episode_reward(
        self, 
        summary: EpisodeSummary, 
        reward: float, 
        components: Dict[str, Any]
    ) -> None:
        """Log episode summary and reward."""
        try:
            win_rate = (summary.wins / summary.totaltrades) if summary.totaltrades > 0 else 0.0
            avg_return = (summary.grossreturnfrac / summary.totaltrades) if summary.totaltrades > 0 else 0.0
            
            msg = (
                f"Episode completed: {summary.totaltrades} trades, "
                f"{win_rate:.1%} win rate, {avg_return:.2%} avg return, "
                f"{summary.maxdrawdownfrac:.2%} max DD, "
                f"Sharpe: {summary.sharpeestimate:.2f if summary.sharpeestimate else 'N/A'}, "
                f"Episode reward: {reward:.1f}"
            )
            
            self.logger.info(msg)
            
            # Send telegram summary for significant episodes
            if self.sendtelegramfn and summary.totaltrades >= 10:
                telegram_msg = (
                    f"📊 Episode Summary\n"
                    f"Trades: {summary.totaltrades} ({win_rate:.1%} wins)\n"
                    f"Avg Return: {avg_return:.2%}\n"
                    f"Max DD: {summary.maxdrawdownfrac:.2%}\n"
                    f"Reward: {reward:.1f}"
                )
                self.sendtelegramfn(telegram_msg)
                
        except Exception as e:
            self.logger.error(f"Error logging episode reward: {e}")


class RewardPersistence:
    """Handles persistent storage of reward events and metrics."""
    
    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        try:
            Path(self.cfg.patheventsjsonl).parent.mkdir(parents=True, exist_ok=True)
            Path(self.cfg.pathtrademetricscsv).parent.mkdir(parents=True, exist_ok=True)
            Path(self.cfg.pathepisodemetricscsv).parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            # Don't crash if directory creation fails
            pass
    
    def appendevent(self, event: Dict[str, Any]) -> None:
        """Append reward event to JSONL file."""
        try:
            event_with_timestamp = {
                **event,
                "timestamp": datetime.now().isoformat(),
                "unix_timestamp": datetime.now().timestamp()
            }
            
            with open(self.cfg.patheventsjsonl, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event_with_timestamp) + '\n')
                
        except Exception as e:
            # Don't crash the trading loop on logging failures
            pass
    
    def appendtrademetrics(self, row: Dict[str, Any]) -> None:
        """Append trade metrics to CSV file."""
        try:
            file_exists = os.path.exists(self.cfg.pathtrademetricscsv)
            
            with open(self.cfg.pathtrademetricscsv, 'a', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'timestamp', 'symbol', 'side', 'entry_price', 'exit_price', 
                    'qty', 'leverage', 'realized_profit_pct', 'fees_usd', 
                    'slippage_usd', 'holding_time_hours', 'reward', 
                    'pnl_points', 'loss_points', 'band_bonus', 'fees_penalty',
                    'sl_violation', 'equity_usd', 'drawdown_frac'
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(row)
                
        except Exception as e:
            # Don't crash the trading loop on logging failures
            pass
    
    def appendepisodemetrics(self, row: Dict[str, Any]) -> None:
        """Append episode metrics to CSV file."""
        try:
            file_exists = os.path.exists(self.cfg.pathepisodemetricscsv)
            
            with open(self.cfg.pathepisodemetricscsv, 'a', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'timestamp', 'episode_id', 'total_trades', 'wins', 'losses',
                    'win_rate', 'gross_return_frac', 'avg_return_per_trade',
                    'max_drawdown_frac', 'sharpe_estimate', 'episode_reward',
                    'banded_profit_points', 'banded_loss_points', 'win_rate_bonus',
                    'sharpe_bonus', 'avg_profit_bonus', 'dd_penalty'
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(row)
                
        except Exception as e:
            # Don't crash the trading loop on logging failures
            pass


def onstep(
    ctx: StepContext, 
    cfg: RewardConfig, 
    adapters: RewardAdapters, 
    persist: RewardPersistence
) -> float:
    """
    Process a single trading step and return reward.
    
    Args:
        ctx: Step context with trade and market data
        cfg: Reward configuration
        adapters: Logging and notification adapters
        persist: Persistence layer for metrics
    
    Returns:
        Calculated reward for the step
    """
    try:
        # Calculate reward
        reward, components = computestepreward(cfg, ctx)
        
        # Log the reward
        adapters.log_step_reward(ctx, reward, components)
        
        # Persist event
        event = {
            "type": "step",
            "symbol": ctx.symbol,
            "side": ctx.side,
            "trade_closed": ctx.tradeclosed,
            "realized_profit_pct": ctx.realizedprofitperc,
            "equity_usd": ctx.equityusd,
            "drawdown_frac": (ctx.peakequityusd - ctx.equityusd) / ctx.peakequityusd if ctx.peakequityusd > 0 else 0.0,
            "reward": reward,
            "components": components
        }
        persist.appendevent(event)
        
        # Persist trade metrics for closed trades
        if ctx.tradeclosed and ctx.realizedprofitperc is not None:
            trade_row = {
                "timestamp": datetime.now().isoformat(),
                "symbol": ctx.symbol,
                "side": ctx.side,
                "entry_price": ctx.entryprice,
                "exit_price": ctx.exitprice,
                "qty": ctx.qty,
                "leverage": ctx.leverage,
                "realized_profit_pct": ctx.realizedprofitperc,
                "fees_usd": ctx.feespaidusd,
                "slippage_usd": ctx.slippageusd,
                "holding_time_hours": ctx.holdingtimeseconds / 3600.0,
                "reward": reward,
                "pnl_points": components.get("pnlpoints", 0),
                "loss_points": components.get("losspoints", 0),
                "band_bonus": components.get("bandbonus", 0),
                "fees_penalty": components.get("feespenalty", 0),
                "sl_violation": components.get("slviolation", 0),
                "equity_usd": ctx.equityusd,
                "drawdown_frac": event["drawdown_frac"]
            }
            persist.appendtrademetrics(trade_row)
        
        return reward
        
    except Exception as e:
        # Don't crash the trading loop - return neutral reward
        adapters.logger.error(f"Error in onstep: {e}")
        return 0.0


def onepisodeend(
    summary: EpisodeSummary, 
    cfg: RewardConfig, 
    adapters: RewardAdapters, 
    persist: RewardPersistence,
    episode_id: str = None
) -> float:
    """
    Process end of trading episode and return episode reward.
    
    Args:
        summary: Episode summary with aggregated metrics
        cfg: Reward configuration
        adapters: Logging and notification adapters
        persist: Persistence layer for metrics
        episode_id: Optional episode identifier
    
    Returns:
        Calculated reward for the episode
    """
    try:
        # Calculate episode reward
        reward, components = computeepisodereward(cfg, summary)
        
        # Log the episode reward
        adapters.log_episode_reward(summary, reward, components)
        
        # Persist episode event
        event = {
            "type": "episode",
            "episode_id": episode_id or f"ep_{int(datetime.now().timestamp())}",
            "total_trades": summary.totaltrades,
            "wins": summary.wins,
            "losses": summary.losses,
            "gross_return_frac": summary.grossreturnfrac,
            "max_drawdown_frac": summary.maxdrawdownfrac,
            "sharpe_estimate": summary.sharpeestimate,
            "reward": reward,
            "components": components
        }
        persist.appendevent(event)
        
        # Persist episode metrics
        win_rate = (summary.wins / summary.totaltrades) if summary.totaltrades > 0 else 0.0
        avg_return = (summary.grossreturnfrac / summary.totaltrades) if summary.totaltrades > 0 else 0.0
        
        episode_row = {
            "timestamp": datetime.now().isoformat(),
            "episode_id": event["episode_id"],
            "total_trades": summary.totaltrades,
            "wins": summary.wins,
            "losses": summary.losses,
            "win_rate": win_rate,
            "gross_return_frac": summary.grossreturnfrac,
            "avg_return_per_trade": avg_return,
            "max_drawdown_frac": summary.maxdrawdownfrac,
            "sharpe_estimate": summary.sharpeestimate,
            "episode_reward": reward,
            "banded_profit_points": summary.bandedprofitpoints,
            "banded_loss_points": summary.bandedlosspoints,
            "win_rate_bonus": components.get("winratebonus", 0),
            "sharpe_bonus": components.get("sharpebonus", 0),
            "avg_profit_bonus": components.get("avgprofitbonus", 0),
            "dd_penalty": components.get("ddpenalty", 0)
        }
        persist.appendepisodemetrics(episode_row)
        
        return reward
        
    except Exception as e:
        # Don't crash the trading loop - return neutral reward
        adapters.logger.error(f"Error in onepisodeend: {e}")
        return 0.0
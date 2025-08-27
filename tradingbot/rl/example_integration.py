"""
Example integration of the Reward Engine with the existing trading bot.
This shows how to integrate the reward system into your existing paper trader or live trader.
"""
import logging
from typing import Optional, Dict, Any

from .rewardconfig import RewardConfig
from .rewardengine import StepContext, EpisodeSummary
from .rewardintegration import RewardAdapters, RewardPersistence, onstep, onepisodeend


class TradingBotRewardIntegration:
    """
    Integration class to add reward system to existing trading bot.
    This can be added to your PaperTrader or live trading system.
    """
    
    def __init__(
        self, 
        logger: logging.Logger,
        telegram_send_fn: Optional[callable] = None,
        reward_config: Optional[RewardConfig] = None
    ):
        """
        Initialize reward system integration.
        
        Args:
            logger: Your existing bot logger
            telegram_send_fn: Optional function to send Telegram messages
            reward_config: Optional custom reward configuration
        """
        self.logger = logger
        self.cfg = reward_config or RewardConfig()
        
        # Initialize reward system components
        self.adapters = RewardAdapters(logger, telegram_send_fn, self.cfg)
        self.persistence = RewardPersistence(self.cfg)
        
        # Tracking state
        self.episode_trades = []
        self.peak_equity = 0.0
        self.rolling_returns = []
        self.consecutive_sl_hits = 0
        
        self.logger.info("Reward system initialized with RL integration")
    
    def on_trade_opened(self, trade_data: Dict[str, Any]) -> None:
        """
        Call this when a new trade is opened.
        
        Args:
            trade_data: Dict with trade information (symbol, side, entry_price, etc.)
        """
        # You can add step reward for trade opening if desired
        # For now, we mainly focus on closed trades
        pass
    
    def on_trade_closed(
        self, 
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        leverage: float = 1.0,
        fees_paid: float = 0.0,
        slippage: float = 0.0,
        holding_time_seconds: float = 0.0,
        current_equity: float = 0.0,
        open_exposure: float = 0.0,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None
    ) -> float:
        """
        Call this when a trade is closed to get the reward.
        
        Returns:
            The calculated reward for this trade
        """
        try:
            # Calculate realized profit percentage
            if side.lower() == 'long':
                profit_frac = (exit_price - entry_price) / entry_price
            else:  # short
                profit_frac = (entry_price - exit_price) / entry_price
            
            # Account for fees and slippage
            cost_frac = (fees_paid + slippage) / (entry_price * quantity)
            realized_profit_pct = profit_frac - cost_frac
            
            # Update tracking state
            self.peak_equity = max(self.peak_equity, current_equity)
            
            # Check for stop loss hit
            if realized_profit_pct < -0.05:  # More than 5% loss
                self.consecutive_sl_hits += 1
            else:
                self.consecutive_sl_hits = 0
            
            # Update rolling returns for Sharpe calculation
            step_return = realized_profit_pct
            self.rolling_returns.append(step_return)
            if len(self.rolling_returns) > 20:  # Keep last 20 returns
                self.rolling_returns.pop(0)
            
            # Create step context
            ctx = StepContext(
                symbol=symbol,
                side=side.lower(),
                entryprice=entry_price,
                exitprice=exit_price,
                midprice=exit_price,  # Use exit price as proxy
                qty=quantity,
                leverage=leverage,
                feespaidusd=fees_paid,
                slippageusd=slippage,
                stoplossprice=stop_loss_price,
                takeprofitprice=take_profit_price,
                holdingtimeseconds=holding_time_seconds,
                tradeclosed=True,
                realizedprofitperc=realized_profit_pct,
                equityusd=current_equity,
                peakequityusd=self.peak_equity,
                openexposureusd=open_exposure,
                consecutivestoplosshits=self.consecutive_sl_hits,
                rollingreturnswindow=self.rolling_returns.copy(),
                dtseconds=60.0  # Assume 1-minute steps
            )
            
            # Calculate reward
            reward = onstep(ctx, self.cfg, self.adapters, self.persistence)
            
            # Store trade for episode calculation
            self.episode_trades.append({
                'symbol': symbol,
                'realized_profit_pct': realized_profit_pct,
                'reward': reward,
                'timestamp': ctx.dtseconds
            })
            
            # Trigger UI update if reward manager is available
            try:
                # This would integrate with your dashboard
                if hasattr(self, 'ui_callback'):
                    trade_data = {
                        'symbol': symbol,
                        'side': side,
                        'realizedProfitPct': realized_profit_pct,
                        'reward': reward,
                        'entryPrice': entry_price,
                        'exitPrice': exit_price,
                        'qty': quantity,
                        'leverage': leverage,
                        'components': {}  # Would need to extract from reward calculation
                    }
                    self.ui_callback(trade_data)
            except Exception as e:
                self.logger.warning(f"Failed to update UI with reward: {e}")
            
            return reward
            
        except Exception as e:
            self.logger.error(f"Error calculating trade reward: {e}")
            return 0.0
    
    def on_episode_end(self, episode_id: Optional[str] = None) -> float:
        """
        Call this when a trading episode (session) ends.
        
        Args:
            episode_id: Optional identifier for the episode
            
        Returns:
            The calculated episode reward
        """
        try:
            if not self.episode_trades:
                return 0.0
            
            # Calculate episode metrics
            total_trades = len(self.episode_trades)
            wins = sum(1 for t in self.episode_trades if t['realized_profit_pct'] > 0)
            losses = total_trades - wins
            
            gross_return = sum(t['realized_profit_pct'] for t in self.episode_trades)
            
            # Calculate max drawdown (simplified)
            running_equity = 1.0
            peak = 1.0
            max_dd = 0.0
            
            for trade in self.episode_trades:
                running_equity *= (1 + trade['realized_profit_pct'])
                peak = max(peak, running_equity)
                dd = (peak - running_equity) / peak
                max_dd = max(max_dd, dd)
            
            # Estimate Sharpe from trade returns
            if len(self.episode_trades) > 1:
                import statistics
                returns = [t['realized_profit_pct'] for t in self.episode_trades]
                try:
                    mean_ret = statistics.mean(returns)
                    std_ret = statistics.stdev(returns)
                    sharpe = mean_ret / std_ret if std_ret > 0 else None
                except:
                    sharpe = None
            else:
                sharpe = None
            
            # Calculate banded points from individual rewards
            profit_points = sum(t['reward'] for t in self.episode_trades if t['reward'] > 0)
            loss_points = sum(t['reward'] for t in self.episode_trades if t['reward'] < 0)
            
            # Create episode summary
            summary = EpisodeSummary(
                totaltrades=total_trades,
                wins=wins,
                losses=losses,
                grossreturnfrac=gross_return,
                maxdrawdownfrac=max_dd,
                sharpeestimate=sharpe,
                equitypath=[],  # Would need to track this
                bandedprofitpoints=profit_points,
                bandedlosspoints=loss_points
            )
            
            # Calculate episode reward
            episode_reward = onepisodeend(summary, self.cfg, self.adapters, self.persistence, episode_id)
            
            # Reset for next episode
            self.episode_trades = []
            
            return episode_reward
            
        except Exception as e:
            self.logger.error(f"Error calculating episode reward: {e}")
            return 0.0
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics for display."""
        if not self.episode_trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_reward_points': 0.0,
                'avg_reward_per_trade': 0.0
            }
        
        total_trades = len(self.episode_trades)
        wins = sum(1 for t in self.episode_trades if t['realized_profit_pct'] > 0)
        win_rate = wins / total_trades if total_trades > 0 else 0.0
        
        total_reward = sum(t['reward'] for t in self.episode_trades)
        avg_reward = total_reward / total_trades if total_trades > 0 else 0.0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_reward_points': total_reward,
            'avg_reward_per_trade': avg_reward,
            'consecutive_sl_hits': self.consecutive_sl_hits
        }
    
    def reset_state(self) -> None:
        """Reset tracking state (useful between episodes)."""
        self.episode_trades = []
        self.peak_equity = 0.0
        self.rolling_returns = []
        self.consecutive_sl_hits = 0


# Example usage in your paper trader:
"""
# In your PaperTrader class:

class PaperTrader:
    def __init__(self):
        # Your existing initialization
        self.logger = logging.getLogger(__name__)
        
        # Add reward integration
        self.reward_system = TradingBotRewardIntegration(
            logger=self.logger,
            telegram_send_fn=self.send_telegram_message  # if you have this
        )
    
    def close_position(self, symbol, exit_price):
        # Your existing trade closing logic
        
        # After closing the trade, calculate reward
        reward = self.reward_system.on_trade_closed(
            symbol=symbol,
            side=self.positions[symbol]['side'],
            entry_price=self.positions[symbol]['entry_price'],
            exit_price=exit_price,
            quantity=self.positions[symbol]['quantity'],
            leverage=self.positions[symbol].get('leverage', 1.0),
            fees_paid=self.calculate_fees(),
            holding_time_seconds=time.time() - self.positions[symbol]['entry_time'],
            current_equity=self.get_portfolio_value(),
            open_exposure=self.get_total_exposure()
        )
        
        self.logger.info(f"Trade reward: {reward:.2f} points")
        
        # Optionally trigger UI update
        if hasattr(self, 'websocket_manager'):
            trade_data = {
                'symbol': symbol,
                'side': self.positions[symbol]['side'],
                'realizedProfitPct': profit_pct,
                'reward': reward,
                # ... other trade data
            }
            # Send to dashboard via websocket
            self.websocket_manager.broadcast('trade_reward', trade_data)
"""
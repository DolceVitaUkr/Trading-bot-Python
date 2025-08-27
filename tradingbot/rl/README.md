# Trading Bot Reward & Penalty Engine

A comprehensive reinforcement learning reward system designed to integrate seamlessly with your existing trading bot infrastructure.

## 🎯 Overview

This reward engine provides deterministic, configurable reward calculations for reinforcement learning applications in algorithmic trading. It implements the exact specifications you requested with zero breaking changes to your existing bot.

## 📦 Components

### Core Files

1. **`rewardconfig.py`** - Typed configuration with environment variable overrides
2. **`rewardengine.py`** - Pure deterministic reward calculation core
3. **`rewardintegration.py`** - Safe integration adapters with logging/persistence
4. **`rewardmetricsschema.json`** - JSON schema for persisted reward events
5. **`example_integration.py`** - Complete integration example for your existing bot
6. **`tests/testrewardengine.py`** - Comprehensive test suite (25 tests, all passing)

### UI Components

- **Reward Modal** - Detailed breakdown of each trade's reward components
- **Reward History Sidebar** - Scrollable history with quick stats
- **Responsive Design** - Works on desktop and mobile
- **Auto-close & Manual Controls** - 8-second auto-close, ESC key, click outside

## 🔧 Quick Start

### 1. Basic Integration

```python
from tradingbot.rl.example_integration import TradingBotRewardIntegration

# In your PaperTrader or LiveTrader class
class YourTradingBot:
    def __init__(self):
        self.reward_system = TradingBotRewardIntegration(
            logger=self.logger,
            telegram_send_fn=self.send_telegram  # Optional
        )
    
    def on_trade_closed(self, symbol, side, entry_price, exit_price, quantity):
        reward = self.reward_system.on_trade_closed(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            leverage=self.get_leverage(),
            fees_paid=self.calculate_fees(),
            holding_time_seconds=self.get_holding_time(),
            current_equity=self.get_portfolio_value(),
            open_exposure=self.get_total_exposure()
        )
        
        self.logger.info(f"Trade reward: {reward:.2f} points")
```

### 2. UI Integration

The reward display is automatically available in your dashboard. When trades close:

```javascript
// In your trade completion handler
if (window.rewardManager) {
    window.rewardManager.onTradeComplete({
        symbol: 'BTCUSDT',
        side: 'long',
        realizedProfitPct: 0.15,
        reward: 12.5,
        components: {
            pnlpoints: 5.0,
            bandbonus: 3.0,
            feespenalty: 2.5,
            sharpebonus: 2.0
        }
    });
}
```

## 🎖️ Reward Structure

### Profit Bands (Implemented exactly as requested)
- **11-19% profit**: +1 point
- **20-30% profit**: +5 points  
- **31-50% profit**: +10 points
- **51-60% profit**: +20 points
- **61-80% profit**: +70 points
- **81-99% profit**: +200 points
- **≥100% profit**: +500 points

### Loss Penalties
- **Linear penalty**: -5 points per 1% loss
- **Trade bonus**: +3 points only if realized profit ≥ 10%

### Risk Management
- **Stop Loss Violations**: Penalty when loss exceeds 10%
- **Exposure Penalties**: When position size > 40% of equity
- **Leverage Penalties**: Tiered leverage limits based on account size
- **Drawdown Monitoring**: Soft (15%) and hard (20%) limits
- **Kill Switch**: Activates at 30% drawdown or 5+ consecutive SL hits

### Performance Shaping
- **Sharpe Bonus**: Rewards for risk-adjusted returns > 1.0
- **Win Rate Bonus**: Rewards when win rate ≥ 75%
- **Holding Time Decay**: Penalty for extended position holding
- **Consecutive SL Penalty**: Escalating penalties for repeated losses

## 📊 Data Persistence

### JSONL Event Log
All reward events are logged with full context:
```json
{
  "type": "step",
  "timestamp": "2024-01-01T12:00:00Z",
  "symbol": "BTCUSDT",
  "reward": 12.5,
  "components": { "pnlpoints": 5.0, "bandbonus": 3.0 }
}
```

### CSV Metrics
- **Trade Metrics**: Per-trade analysis with all reward components
- **Episode Metrics**: Session-level performance aggregations
- **Rolling Statistics**: Sharpe ratio, drawdown, win rates

## ⚙️ Configuration

All parameters are configurable via environment variables or direct config:

```python
from tradingbot.rl.rewardconfig import RewardConfig

config = RewardConfig(
    targetwinrate=0.80,        # 80% target win rate
    targetsharpe=2.5,          # Higher Sharpe target
    maxdrawdownfracsoft=0.12,  # Tighter drawdown limit
    # ... all other parameters
)
```

### Environment Variables
```bash
TARGET_WIN_RATE=0.80
TARGET_SHARPE=2.5
MAX_DD_SOFT=0.12
LOSS_POINTS_PER_PERC=-10.0
# See rewardconfig.py for complete list
```

## 🧪 Testing

Comprehensive test suite with 25 tests covering:
- Core reward calculations
- Edge cases and error handling  
- Integration components
- Extreme value handling
- File persistence operations

```bash
cd Trading-bot-Python
python tests/testrewardengine.py
# All tests passing ✓
```

## 🔐 Security & Reliability

- **No Network Dependencies**: All operations are local
- **Exception Handling**: Never crashes your trading loop
- **File Safety**: Graceful degradation if file operations fail
- **Memory Efficient**: Rolling windows with configurable limits
- **Deterministic**: Same inputs always produce same outputs

## 📱 UI Features

### Reward Modal
- **Instant Feedback**: Shows immediately after significant trades
- **Detailed Breakdown**: Separate sections for rewards vs penalties
- **Warning System**: Highlights risk violations and errors
- **Auto-Dismiss**: 8-second timer with manual controls

### History Sidebar
- **Scrollable History**: Last 50 rewards with local storage
- **Quick Stats**: Running averages and totals
- **Click-to-Details**: Tap any historical reward for full breakdown
- **Performance Indicators**: Color-coded positive/negative rewards

### Responsive Design
- **Mobile Optimized**: Full-screen modal on small devices
- **Touch Friendly**: Large touch targets and smooth animations
- **Accessibility**: Proper contrast and keyboard navigation

## 🔗 Integration Points

### Paper Trader Integration
```python
# In your paper_trader.py close_position method:
reward = self.reward_system.on_trade_closed(
    symbol=symbol,
    side=position['side'],
    entry_price=position['entry_price'],
    exit_price=current_price,
    quantity=position['quantity'],
    current_equity=self.portfolio_value,
    open_exposure=self.total_open_exposure
)
```

### Live Trading Integration
```python
# Same interface works for live trading
# Add to your existing trade close logic
reward = self.reward_system.on_trade_closed(...)
```

### Dashboard Integration
```javascript
// Already integrated in dashboard.js
// RewardDisplayManager automatically handles UI updates
```

## 📈 Performance Metrics

The system tracks and displays:
- **Individual Trade Rewards**: Detailed component breakdown
- **Episode Performance**: Win rates, Sharpe ratios, max drawdown
- **Risk Monitoring**: Real-time limit violations
- **Rolling Statistics**: Smoothed performance indicators

## 🚀 Production Ready

- **Zero Downtime Integration**: Drop-in components
- **Configurable Paths**: Customize data storage locations
- **Telegram Integration**: Optional notification system
- **Logging Integration**: Uses your existing logger
- **Error Recovery**: Graceful handling of all failure modes

## 🔄 Usage Examples

See `example_integration.py` for complete working examples showing integration with:
- Paper trading systems
- Live trading systems  
- Dashboard updates
- Telegram notifications
- Data persistence
- Error handling

The reward system is now ready for production use with your existing trading bot! 🎉
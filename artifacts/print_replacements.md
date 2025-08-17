# Print Statement Replacement Report

Generated: 2025-08-17 14:46:08
Mode: Analysis

## Summary

Total print statements found: 41

**By log level:**
- error: 10
- info: 29
- warning: 2

Files affected: 13

## Detailed Replacements

### Data_Registry.py

**Add import:**
```python
from modules.Logger_Config import get_logger
logger = get_logger(__name__)
```

**Replacements:**

**Line 294** (info level):
```python
# Before:
print("=== Data Registry Demo ===")
# After:
logger.info('=== Data Registry Demo ===')
```

**Line 295** (info level):
```python
# Before:
print(f"Historical BTCUSDT 5m: {registry.get_historical_data_path('bybit', 'BTCUSDT', '5m')}")
# After:
logger.info(f'...')
```

**Line 296** (info level):
```python
# Before:
print(f"Main branch metrics: {registry.get_metrics_path('main', 'paper')}")
# After:
logger.info(f'...')
```

**Line 297** (info level):
```python
# Before:
print(f"Model path: {registry.get_model_path('main', 'rl_model')}")
# After:
logger.info(f'...')
```

**Line 298** (info level):
```python
# Before:
print(f"Log path: {registry.get_log_path('main', 'paper', 'decisions')}")
# After:
logger.info(f'...')
```

**Line 299** (info level):
```python
# Before:
print(f"Available branches: {registry.get_branch_list()}")
# After:
logger.info(f'...')
```

**Line 300** (info level):
```python
# Before:
print(f"Disk usage: {registry.get_disk_usage_report()}")
# After:
logger.info(f'...')
```

### Self_test/test.py

**Add import:**
```python
from modules.Logger_Config import get_logger
logger = get_logger(__name__)
```

**Replacements:**

**Line 144** (info level):
```python
# Before:
print("\n".join(lines))
# After:
logger.info('\n'.join(lines))
```

**Line 145** (info level):
```python
# Before:
print(f"(report saved to {report_path})")
# After:
logger.info(f'...')
```

### adapters/composite_wallet.py

**Add import:**
```python
from modules.Logger_Config import get_logger
logger = get_logger(__name__)
```

**Replacements:**

**Line 29** (error level):
```python
# Before:
print(f"Error fetching equity from wallet {type(wallet).__name__}: {e}")
# After:
logger.error(f'...')
```

### adapters/ibkr_exec.py

**Add import:**
```python
from modules.Logger_Config import get_logger
logger = get_logger(__name__)
```

**Replacements:**

**Line 62** (warning level):
```python
# Before:
print(f"Warning: No minimum notional defined for {symbol}. Allowing trade.")
# After:
logger.warning(f'...')
```

**Line 79** (info level):
```python
# Before:
print(f"Order for {symbol} blocked: {reason}")
# After:
logger.info(f'...')
```

**Line 89** (info level):
```python
# Before:
print(f"Simulating bracket order for {symbol} with SL={stop_loss_price}, TP={take_profit_price}")
# After:
logger.info(f'...')
```

**Line 125** (error level):
```python
# Before:
print(f"Error placing order for {symbol}: {e}")
# After:
logger.error(f'...')
```

### adapters/ibkr_market.py

**Add import:**
```python
from modules.Logger_Config import get_logger
logger = get_logger(__name__)
```

**Replacements:**

**Line 84** (info level):
```python
# Before:
print(f"No historical data returned for {symbol}.")
# After:
logger.info(f'...')
```

**Line 102** (error level):
```python
# Before:
print(f"Error fetching historical data for {symbol}: {e}")
# After:
logger.error(f'...')
```

**Line 130** (error level):
```python
# Before:
print(f"Error fetching ticker for {symbol}: {e}")
# After:
logger.error(f'...')
```

### adapters/news_rss.py

**Add import:**
```python
from modules.Logger_Config import get_logger
logger = get_logger(__name__)
```

**Replacements:**

**Line 45** (error level):
```python
# Before:
print(f"Error fetching RSS feed {feed_url}: {e}")
# After:
logger.error(f'...')
```

**Line 84** (info level):
```python
# Before:
print(f"Detected new macro event for {currency}: {entry.title}")
# After:
logger.info(f'...')
```

**Line 95** (info level):
```python
# Before:
print(f"Blocking {symbol} due to recent event: '{event_title}'")
# After:
logger.info(f'...')
```

### adapters/null_adapters.py

**Add import:**
```python
from modules.Logger_Config import get_logger
logger = get_logger(__name__)
```

**Replacements:**

**Line 13** (info level):
```python
# Before:
print("NullMarketData: candles() called, returning empty DataFrame.")
# After:
logger.info('NullMarketData: candles() called, returning empty DataFrame.')
```

**Line 17** (info level):
```python
# Before:
print("NullMarketData: ticker() called, returning empty dict.")
# After:
logger.info('NullMarketData: ticker() called, returning empty dict.')
```

**Line 21** (info level):
```python
# Before:
print("NullMarketData: volume_24h() called, returning 0.")
# After:
logger.info('NullMarketData: volume_24h() called, returning 0.')
```

**Line 28** (info level):
```python
# Before:
print(f"NullExecution: place_order({symbol}, {side}, {qty}) called, doing nothing.")
# After:
logger.info(f'...')
```

**Line 32** (info level):
```python
# Before:
print("NullExecution: positions() called, returning empty list.")
# After:
logger.info('NullExecution: positions() called, returning empty list.')
```

**Line 39** (info level):
```python
# Before:
print("NullWalletSync: subledger_equity() called, returning zero balances.")
# After:
logger.info('NullWalletSync: subledger_equity() called, returning zero balances.')
```

**Line 46** (info level):
```python
# Before:
print("NullNewsFeed: sentiment() called, returning neutral sentiment.")
# After:
logger.info('NullNewsFeed: sentiment() called, returning neutral sentiment.')
```

**Line 50** (info level):
```python
# Before:
print("NullNewsFeed: macro_blockers() called, returning no blockers.")
# After:
logger.info('NullNewsFeed: macro_blockers() called, returning no blockers.')
```

**Line 57** (info level):
```python
# Before:
print(f"NullValidationRunner: approved({strategy_id}, {market}) called, returning approved.")
# After:
logger.info(f'...')
```

### adapters/wallet_bybit.py

**Add import:**
```python
from modules.Logger_Config import get_logger
logger = get_logger(__name__)
```

**Replacements:**

**Line 70** (error level):
```python
# Before:
print(f"Exception fetching Bybit balance for {account_type}: {e}")
# After:
logger.error(f'...')
```

**Line 68** (error level):
```python
# Before:
print(f"HTTP Error fetching Bybit balance for {account_type}: {response.status}")
# After:
logger.error(f'...')
```

**Line 66** (error level):
```python
# Before:
print(f"Bybit API error for {account_type}: {data.get('retMsg')}")
# After:
logger.error(f'...')
```

### managers/kill_switch.py

**Add import:**
```python
from modules.Logger_Config import get_logger
logger = get_logger(__name__)
```

**Replacements:**

**Line 69** (info level):
```python
# Before:
print(f"!!! KILL SWITCH TRIGGERED for {scope_key} due to rule '{rule}' !!!")
# After:
logger.info(f'...')
```

**Line 106** (info level):
```python
# Before:
print(f"Auto-rearming kill switch for {scope_key}.")
# After:
logger.info(f'...')
```

**Line 48** (info level):
```python
# Before:
print(f"Kill switch for {key} is active from previous state.")
# After:
logger.info(f'...')
```

### managers/sizer.py

**Add import:**
```python
from modules.Logger_Config import get_logger
logger = get_logger(__name__)
```

**Replacements:**

**Line 38** (warning level):
```python
# Before:
print(f"Warning: No subledger mapping for asset class '{asset_class}'. Sizing will be zero.")
# After:
logger.warning(f'...')
```

**Line 45** (error level):
```python
# Before:
print(f"Warning: No equity available in subledger '{subledger}'. Cannot size order.")
# After:
logger.error(f'...')
```

**Line 59** (error level):
```python
# Before:
print("Warning: Price is zero or negative. Cannot calculate quantity.")
# After:
logger.error('Warning: Price is zero or negative. Cannot calculate quantity.')
```

### managers/strategy_manager.py

**Add import:**
```python
from modules.Logger_Config import get_logger
logger = get_logger(__name__)
```

**Replacements:**

**Line 31** (info level):
```python
# Before:
print(f"Registering strategy: {meta.strategy_id}")
# After:
logger.info(f'...')
```

### managers/validation_manager.py

**Add import:**
```python
from modules.Logger_Config import get_logger
logger = get_logger(__name__)
```

**Replacements:**

**Line 68** (info level):
```python
# Before:
print(f"Updated trade count for {product}: {self.trade_counts[product]}")
# After:
logger.info(f'...')
```

### modules/exchange.py

**Replacements:**

**Line 105** (info level):
```python
# Before:
print("load_markets called")
# After:
logger.info('load_markets called')
```


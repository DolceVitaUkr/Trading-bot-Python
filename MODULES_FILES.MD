# MODULES_FILES.MD

## Project Structure Manifest

### Root Files
- **main.py**  
  - Async entry point with graceful shutdown  
  - Manages component lifecycle and UI integration  
  - Supports both simulation and live trading modes

- **config.py**  
  - Centralized configuration with environment variables  
  - Contains parameters for:  
    - Exchange API credentials  
    - Risk management thresholds  
    - Simulation parameters  
    - Reward system settings  
    - UI/UX preferences  
    - Optimization constraints

- **requirements.txt**  
  - Lists dependencies including:  
    - Async libraries (aiohttp, asyncio)  
    - Data processing (pandas, numpy)  
    - ML frameworks (scikit-learn, tensorflow)  
    - Exchange integration (ccxt)

### Core Modules
- **exchange.py**  
  - Unified interface for ByBit API (REST/WebSocket)  
  - Supports both live and simulated trading  
  - Implements rate limiting and retry logic

- **telegram_bot.py**  
  - Async notification system with message queue  
  - Supports both synchronous and async contexts  
  - Implements message formatting templates

- **ui.py**  
  - Interactive terminal/web interface  
  - Real-time visualization of:  
    - Portfolio performance  
    - Market depth charts  
    - Strategy metrics  
  - Mode toggling (simulation/live)

### Data Management
- **data_manager.py**  
  - Historical data backfilling  
  - OHLCV data normalization  
  - Candle stick pattern recognition

- **top_pairs.py**  
  - Dynamic pair selection system  
  - Maintains:  
    - Active trading pairs from exchange  
    - Cached pair list with TTL  
    - Hardcoded fallback pairs  
  - Pair ranking based on liquidity metrics

### Trading Logic
- **trade_executor.py**  
  - Unified order execution interface  
  - Simulated order matching engine  
  - Real order validation system  
  - Integrated Telegram notifications

- **trade_simulator.py**  
  - Async backtesting engine  
  - Market replay functionality  
  - Monte Carlo simulation support  
  - Detailed performance analytics

### Risk & Analytics
- **risk_management.py**  
  - Dynamic position sizing  
  - Volatility-adjusted stop loss  
  - Correlation matrix analysis  
  - Drawdown circuit breakers

- **reward_system.py**  
  - Multi-factor scoring system  
  - Time-weighted performance metrics  
  - Behavioral penalty system  
  - Skill-based ranking

### AI/ML Components
- **self_learning.py**  
  - Reinforcement learning core  
  - Strategy pattern recognition  
  - Anomaly detection system  
  - Live model retraining

- **parameter_optimization.py**  
  - Multi-objective optimization  
  - Genetic algorithm implementation  
  - Walk-forward testing  
  - Parameter sensitivity analysis

### Support Modules
- **error_handler.py**  
  - Hierarchical error classification  
  - Automated recovery system  
  - Error context preservation  
  - Fail-safe mechanisms

- **technical_indicators.py**  
  - 50+ implemented indicators  
  - Custom indicator creation  
  - Multi-timeframe analysis  
  - Market regime detection

### Infrastructure
- **logs/**  
  - Rotating log files  
  - Structured logging format  
  - Performance audit trails

- **data/**  
  - Tick-by-tick storage  
  - Compressed OHLCV archives  
  - Feature engineering datasets

- **utils/utilities.py**  
  - Consolidated utilities for:  
    - File system operations  
    - Time conversions  
    - Retry logic  
    - Logging configuration  
    - Async utilities  
  - Replaces old helper.py and logger_util.py

## Execution Flow
```mermaid
graph TD
    A[Main] --> B[Config Loader]
    A --> C[Exchange Interface]
    A --> D[Data Pipeline]
    C --> E[Live Trading]
    C --> F[Simulation Engine]
    D --> G[Market Data]
    D --> H[Historical Data]
    E --> I[Risk System]
    F --> I
    I --> J[Order Execution]
    J --> K[Performance Tracking]
    K --> L[AI Retraining]
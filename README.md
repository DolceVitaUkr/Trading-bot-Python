# Modular Trading Bot

This repository contains a modular trading bot designed for both cryptocurrency and Forex trading. The bot is built with a flexible, interface-based architecture that allows for hot-swapping components and features at runtime through environment variables.

## Features

- **Modular Architecture**: Core logic is decoupled from specific exchange/broker implementations through a set of `Protocol` interfaces.
- **Hot-Swappable Components**: Enable or disable major features like Forex trading and News analysis without code changes.
- **Multi-Venue Support**: Includes adapters for Interactive Brokers (Forex) and Bybit (Crypto wallet), with a structure to easily add more.
- **Risk Management**:
    - **Wallet-aware Sizer**: Sizes trades based on the equity of specific sub-ledgers (FX, SPOT, FUTURES).
    - **Kill Switch**: Automatically halts trading for a specific asset class/venue if risk rules (e.g., daily loss limit) are breached.
- **Strategy Governance**: A `ValidationManager` checks strategy performance against predefined thresholds before allowing live execution.
- **Detailed Logging**: Every trading decision is logged to a `decision_traces.jsonl` file for full transparency and debugging.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies

Install all the required Python packages from `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory by copying the example file.

```bash
cp .env.example .env
```

Now, edit the `.env` file with your specific configuration:

```ini
# --- Feature Flags ---
# Set to 1 to enable, 0 to disable
ENABLE_NEWS=0
ENABLE_FOREX=0

# --- IBKR Connection Parameters ---
# For TWS or Gateway. 7497 is default for paper TWS.
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=1

# --- Bybit API Credentials ---
# Required for Crypto wallet sync
BYBIT_API_KEY=your_bybit_api_key
BYBIT_API_SECRET=your_bybit_api_secret
```

### 5. IBKR Setup (for Forex Trading)

To use the Forex features, you must have Interactive Brokers Trader Workstation (TWS) or IB Gateway running.

1.  **Install TWS or Gateway**: Download from the [Interactive Brokers website](https://www.interactivebrokers.com/en/index.php?f=14099).
2.  **Enable API Access**:
    - In TWS, go to `File > Global Configuration > API > Settings`.
    - Check `Enable ActiveX and Socket Clients`.
    - Make note of the `Socket port` number (it should match `IBKR_PORT` in your `.env` file).
    - Add `127.0.0.1` to the `Trusted IP Addresses`.
3.  **Use a Paper Trading Account**: For development and training, log into TWS/Gateway using your **Paper Trading Account** credentials. The bot is configured by default to connect to the paper trading port (`7497`).
4.  **Market Data Subscriptions**: Ensure your paper account has the necessary market data subscriptions for the Forex pairs you intend to trade. The bot will fail to fetch data without them.

## How to Run the Bot

Once the setup is complete, you can run the bot from the root directory.

### Default Mode (Crypto)

With `ENABLE_FOREX=0` in your `.env` file, the bot will run in its default crypto mode. It will use the `Null` adapters for market data and execution but can be configured to use real crypto exchange adapters.

```bash
python main.py
```

### Forex Training Mode

To enable the IBKR integration for Forex paper trading:

1.  Make sure TWS or Gateway is running and you are logged into your paper account.
2.  Set `ENABLE_FOREX=1` in your `.env` file.

```ini
# .env file
ENABLE_FOREX=1
```

3.  Run the bot:

```bash
python main.py
```

The bot will now connect to IBKR, and you will see logs related to Forex data fetching and order placement (if triggered).

### Enabling News

To enable the news feed adapter, set `ENABLE_NEWS=1`. The bot will then fetch news from RSS feeds and can block trades based on macro events.

```ini
# .env file
ENABLE_NEWS=1
```

## Project Structure

- `core/`: Contains the core `interfaces.py` (protocols) and `schemas.py` (Pydantic models).
- `adapters/`: Contains all implementations of the core interfaces for specific services (IBKR, Bybit, News, etc.).
- `managers/`: Contains the high-level business logic components (Sizer, KillSwitch, ValidationManager).
- `state/`: Directory where the bot stores its state in `.jsonl` files (traces, events).
- `main.py`: The main orchestrator and entry point of the application.
- `requirements.txt`: Project dependencies.
- `.env.example`: Example environment configuration.
- `README.md`: This file.

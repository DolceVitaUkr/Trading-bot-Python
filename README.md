# Modular Trading Bot

This repository contains a modular trading bot designed for both cryptocurrency and Forex trading. The bot is built with a flexible, interface-based architecture that allows for hot-swapping components and features at runtime through environment variables.

## Features

- **Modular Architecture**: Core logic is decoupled from specific exchange/broker implementations through a set of `Protocol` interfaces.
- **Hot-Swappable Components**: Enable or disable major features and product pipelines without code changes.
- **Multi-Venue Support**: Features a robust, institutional-grade Interactive Brokers integration for Forex Spot and Options, alongside existing Bybit capabilities for Crypto.
- **Product-Centric Pipelines**: Run independent strategies for different products (e.g., `FOREX_SPOT`, `FOREX_OPTIONS`, `CRYPTO_SPOT`) with separate data, models, and validation thresholds.
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
# --- Product & Feature Flags ---
# Comma-separated list of products to run.
# Supported: CRYPTO_SPOT, FOREX_SPOT, FOREX_OPTIONS
PRODUCTS_ENABLED="CRYPTO_SPOT,FOREX_SPOT"

# --- IBKR Configuration ---
# Set to "paper" or "live"
IBKR_API_MODE="paper"
# If True, blocks all order placement calls. Set to False for live trading.
TRAINING_MODE="True"
# TWS/Gateway socket connection settings
IBKR_TWS_HOST="127.0.0.1"
IBKR_TWS_PORT="7497" # Default for paper TWS, 7496 for live
IBKR_TWS_CLIENT_ID="1"

# Client Portal API (for contract search, etc.)
# See IBKR Setup section for how to run the gateway.
IBKR_CPAPI_GATEWAY_URL="https://localhost:5000"

# --- Bybit API Credentials ---
BYBIT_API_KEY="your_bybit_api_key"
BYBIT_API_SECRET="your_bybit_api_secret"

# --- Other Feature Flags ---
ENABLE_NEWS="0"
```

### 5. IBKR Setup

To use the IBKR integration, you must have Interactive Brokers Trader Workstation (TWS) or IB Gateway running.

1.  **Install TWS or Gateway**: Download from the [Interactive Brokers website](https://www.interactivebrokers.com/en/index.php?f=14099).
2.  **Enable API Access (in TWS/Gateway)**:
    - Go to `File > Global Configuration > API > Settings`.
    - Check `Enable ActiveX and Socket Clients`.
    - **Crucially, for the initial training phase, check `Read-Only API`**. This provides a server-side safety lock to prevent any order modifications.
    - Make note of the `Socket port` number (it should match `IBKR_TWS_PORT` in your `.env` file).
    - Add `127.0.0.1` to the `Trusted IP Addresses`.
3.  **Run the Client Portal API Gateway (Optional but Recommended)**:
    - The bot uses the Client Portal API for robust contract searches (especially for options). To enable this, you need to run the local gateway.
    - Follow the instructions on the [Interactive Brokers GitHub page](https://interactivebrokers.github.io/) to download and run the `run.sh` (or `run.bat`) script for the `clientportal.gw`.
4.  **Use a Paper Trading Account**: For all development and training, log into TWS/Gateway using your **Paper Trading Account** credentials. The bot validates this and will raise an error if `IBKR_API_MODE` is `"paper"` but it detects a live account.
5.  **Market Data Subscriptions**: Ensure your paper account has the necessary market data subscriptions. The bot will perform a pre-flight check and fail if data is missing.
    - For **Forex Spot**, you need "Forex" data.
    - For **Forex Options**, you need "US Options" data (as FX options are often cleared through US exchanges).

## Running the Bot

The bot now operates based on the `PRODUCTS_ENABLED` variable in your `.env` file. It will initialize the required connections and run the corresponding training pipelines for each enabled product.

### Example: Running Crypto and Forex Spot

1.  Ensure your Bybit API keys are set in `.env`.
2.  Ensure TWS or Gateway is running and you are logged into your paper account.
3.  Set the products in your `.env` file:

```ini
PRODUCTS_ENABLED="CRYPTO_SPOT,FOREX_SPOT"
TRAINING_MODE="True"
```

4.  Run the bot:

```bash
python main.py
```

The bot will now connect to both Bybit (for wallet sync) and IBKR. It will run the training pipeline for `CRYPTO_SPOT` (using Bybit data if configured) and `FOREX_SPOT` (using IBKR data). All order placements will be blocked because `TRAINING_MODE` is on.

### Example: Running Forex Options Only

1.  Ensure TWS or Gateway is running.
2.  Set the products in your `.env` file:

```ini
PRODUCTS_ENABLED="FOREX_OPTIONS"
TRAINING_MODE="True"
```

3.  Run the bot:

```bash
python main.py
```

The bot will now connect to IBKR and start the training pipeline for Forex options, which involves fetching option chains and greeks.

## Project Structure

- `core/`: Contains the core `interfaces.py` (protocols) and `schemas.py` (Pydantic models).
- `adapters/`: Contains all implementations of the core interfaces for specific services (IBKR, Bybit, News, etc.).
- `managers/`: Contains the high-level business logic components (Sizer, KillSwitch, ValidationManager).
- `state/`: Directory where the bot stores its state in `.jsonl` files (traces, events).
- `main.py`: The main orchestrator and entry point of the application.
- `requirements.txt`: Project dependencies.
- `.env.example`: Example environment configuration.
- `README.md`: This file.

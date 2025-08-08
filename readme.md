# Self-Learning Crypto Trading Bot (Bybit, 15m)

Trains on **live market data** with **paper fills**, auto-promotes to **live trading** once KPIs pass, then **keeps learning while live** using safe **exploration (paper) trades**. Adds a **20-minute “Armed Watchlist”** to monitor near-setups and only fire when confirmations appear.

---

## Core Principles
- **Paper-on-Live-Data First:** Train only on real 15m candles; simulate fills until promotion gates pass.
- **Auto Promotion / De-promotion:** Strict KPI thresholds to switch between paper and live.
- **Continual Learning (Live):** While trading real orders, the bot still trains online and runs **exploration paper trades**.
- **Top-N Pair Rotation:** Cycle the most liquid/volatile USDT pairs (default N=30) with per-pair cooldown.
- **Armed Watchlist (20m Monitor):** Near-setups get **armed** for up to 20 minutes; trade only if confirmations complete.
- **Bybit-Safe:** Throttled, precise, compliant API usage with symbol filters and idempotent order keys.
- **Risk First:** R:R gate (≥3:1), per-trade & portfolio limits, daily loss stops, circuit breakers.

---

## Targets & Guardrails
- **Timeframe:** `15m`
- **Rotation:** `N=30` USDT pairs (config), per-pair `cooldown=60m`
- **Promotion (Paper → Live)** — min **150 trades** in last **14 days** and:
  - Win% ≥ **75%**, *or* Win% ≥ **60%** with **R:R ≥ 3.0**
  - Rolling **Sharpe ≥ 2.0**
  - Avg profit/trade: **Scalp 1–3%**, **Swing 10–30%**
  - **Max drawdown ≤ 15%**
- **De-promotion (Live → Paper):** Win% < **60%** over last **20 trades** *or* Max DD > **20%**
- **Exploration in Live:** About **1 paper exploration trade per 3 live trades** (`EXPLORATION_RATE≈0.25`).
- **Risk Rails:** Daily loss stop, rolling DD stop, consecutive-loss cooldown, precision/minQty validation, R:R gate.

---

## Armed Watchlist (20-Minute Monitor)
- **State Machine:** **Trigger → Arm → Fire → Expire**
  - **Trigger:** Setup is close (e.g., R:R 2.5–3.0, RSI/ADX/volume shaping) but missing final confirmation.
  - **Arm:** Add pair to **Watchlist** with **TTL = 20m** (configurable; ATR-adaptive optional).
  - **Fire:** If confirmations complete within TTL (e.g., candle close/volume/ADX), and **R:R ≥ 3.0** → place trade.
  - **Expire:** If not confirmed by TTL → remove and **cool-off** pair.
- **Limits:** Max **5** armed pairs; recheck every **60–120s** using intrabar/micro-bars.
- **Exploration synergy:** If a near-setup repeatedly expires, allow **paper exploration** to gather learning signals (no capital risk).
- **Telemetry:** Armed→Fired conversion, avg armed time, performance vs non-armed trades.

---

## Folder Layout

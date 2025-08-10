# modules/self_learning.py

import random
import logging
from collections import deque, namedtuple
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import config
from modules.data_manager import DataManager
from modules.top_pairs import TopPairs
from modules.trade_executor import TradeExecutor
from modules.risk_management import RiskManager, RiskViolationError
from modules.technical_indicators import TechnicalIndicators
from modules.reward_system import RewardSystem, calculate_points

Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(getattr(logging, str(getattr(config, "LOG_LEVEL", "INFO")), logging.INFO)
                if isinstance(getattr(config, "LOG_LEVEL", "INFO"), str) else getattr(config, "LOG_LEVEL", logging.INFO))


# ────────────────────────────────────────────────────────────────────────────────
# Replay Buffer
# ────────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ────────────────────────────────────────────────────────────────────────────────
# DQN
# ────────────────────────────────────────────────────────────────────────────────

class DQNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.out = nn.Linear(hidden_dims[2], output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)


# ────────────────────────────────────────────────────────────────────────────────
# Self-Learning Bot (simulation-only execution; live market data via REST)
# ────────────────────────────────────────────────────────────────────────────────

class SelfLearningBot:
    """
    Reinforcement-learning trader that:
      - Pulls market data incrementally (5m/15m) via DataManager
      - Chooses symbols from TopPairs (cached 60 min) or DEFAULT_SYMBOL fallback
      - Trades in SIMULATION mode only (TradeExecutor(simulation_mode=True))
      - Tightens stops and re-evaluates open positions every 1–5 minutes
      - Keeps requests light (append only a few bars per tick)
    """

    def __init__(
        self,
        data_manager: Optional[DataManager] = None,
        reward_system: Optional[RewardSystem] = None,
        risk_manager: Optional[RiskManager] = None,
        *,
        state_size: int = 8,              # includes x-timeframe features
        action_size: int = 6,             # buy/sell/hold/close + extras
        hidden_dims: List[int] = [128, 64, 32],
        batch_size: int = 64,
        gamma: float = 0.99,
        learning_rate: float = 1e-3,
        exploration_max: float = 1.0,
        exploration_min: float = 0.05,
        exploration_decay: float = 0.995,
        memory_size: int = 100_000,
        tau: float = 0.005,
        timeframe_entry: str = "5m",
        timeframe_setup: str = "15m",
        symbol: Optional[str] = None,
    ):
        self.training = True  # UI will flip if needed
        self.timeframe_entry = timeframe_entry
        self.timeframe_setup = timeframe_setup

        # Data / market selection
        self.data_manager = data_manager or DataManager()
        self.top_pairs = TopPairs(exchange_id="bybit", spot=True, cache_minutes=60)
        self.default_symbol = symbol or getattr(config, "DEFAULT_SYMBOL", "BTC/USDT")
        self.current_symbol = self.default_symbol

        # Risk & rewards
        self.reward_system = reward_system or RewardSystem()
        self.risk_manager = risk_manager or RiskManager(
            account_balance=float(getattr(config, "SIMULATION_START_BALANCE", 1000.0))
        )
        # Executor strictly in simulation mode
        self.executor = TradeExecutor(simulation_mode=True)

        # DQN bits
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.tau = tau

        self.epsilon = exploration_max
        self.epsilon_min = exploration_min
        self.epsilon_decay = exploration_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = ReplayBuffer(capacity=memory_size)
        self.policy_net = DQNetwork(state_size, hidden_dims, action_size).to(self.device)
        self.target_net = DQNetwork(state_size, hidden_dims, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.train_steps = 0

        # Execution rhythm control
        self._last_top_pairs_refresh_ts: float = 0.0
        self._last_entry_scan_ts: float = 0.0
        self._last_position_check_ts: float = 0.0
        self.scan_interval_sec = max(60.0, float(getattr(config, "LIVE_LOOP_INTERVAL", 5.0)))   # entries
        self.position_check_sec = 60.0 * 1.0  # re-evaluate open positions at ~1m; adjust to 5m if too chatty

        # Housekeeping
        self.logger = logger

    # ───────────────────────── High-level control ───────────────────────── #

    def refresh_symbol_universe(self) -> None:
        now = datetime.now(timezone.utc).timestamp()
        if now - self._last_top_pairs_refresh_ts < (60 * 60):  # 60 minutes
            return
        try:
            pairs = self.top_pairs.get_top_pairs(max_pairs=getattr(config, "MAX_SIMULATION_PAIRS", 5))
            if pairs:
                # Prefer the first; fallback to default if none
                self.current_symbol = pairs[0]
            else:
                self.current_symbol = self.default_symbol
            self._last_top_pairs_refresh_ts = now
            self.logger.info(f"[Universe] Current symbol set to {self.current_symbol}")
        except Exception as e:
            self.logger.warning(f"[Universe] refresh failed, stick to {self.current_symbol}: {e}")

    def step(self) -> None:
        """
        One outer-loop iteration: refresh data, pick/maintain symbol, act & learn.
        The caller (main loop / scheduler) should invoke this every LIVE_LOOP_INTERVAL seconds.
        """
        # Universe update (hourly)
        self.refresh_symbol_universe()
        sym = self.current_symbol

        # Load incremental data (5m and 15m)
        df5, df15 = self.data_manager.load_dual_timeframe(
            sym, self.timeframe_entry, self.timeframe_setup,
            max_bars_entry=900, max_bars_setup=900, append_increment=5
        )
        if len(df5) == 0 or len(df15) == 0:
            self.logger.info("No Historical Data available for simulation (yet).")
            return

        # Build state from dual timeframe
        state = self._compose_state(df5, df15)

        # Periodic re-eval of open positions (tighter cadence)
        now_ts = datetime.now(timezone.utc).timestamp()
        if now_ts - self._last_position_check_ts >= self.position_check_sec:
            self._manage_open_positions(sym, df5)
            self._last_position_check_ts = now_ts

        # Entry/exit decisions less frequently than position checks
        if now_ts - self._last_entry_scan_ts < self.scan_interval_sec:
            return
        self._last_entry_scan_ts = now_ts

        # Action selection
        action_idx = self._select_action(state)
        action_name = self._action_to_command(action_idx)

        # Execute in simulation mode
        try:
            tr = self._execute_action(sym, action_name, df5)
            # Compute reward (instantaneous; you can expand with next tick PnL)
            reward = float(tr.get("profit") or 0.0)
            ts = datetime.now(timezone.utc)
            next_state = self._compose_state(*self.data_manager.load_dual_timeframe(sym, self.timeframe_entry, self.timeframe_setup))
            done = False

            self.memory.push(state, action_idx, reward, next_state, done)
            self.train_steps += 1
            if self.training:
                self._learn()
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            self.logger.info(f"[Step {self.train_steps}] {sym} action={action_name} reward={reward:.4f} eps={self.epsilon:.3f}")

        except RiskViolationError as re:
            self.logger.info(f"[Risk] blocked: {re}")
        except Exception as e:
            self.logger.exception(f"[Act] failed: {e}")

    # ───────────────────────── Action plumbing ───────────────────────── #

    def _execute_action(self, symbol: str, action_name: str, df5) -> Dict:
        price = float(df5["close"].iloc[-1])
        atr = TechnicalIndicators.atr(
            df5["high"].tolist(), df5["low"].tolist(), df5["close"].tolist(), period=14
        ) or 0.0

        qty = None
        sl_px, tp_px = None, None
        risk_close = False

        if action_name in ("buy", "sell"):
            side = "long" if action_name == "buy" else "short"
            # initial SL/TP via RiskManager using ATR bands & RR
            pos = self.risk_manager.calculate_position_size(
                symbol=symbol,
                side=side,  # type: ignore[arg-type]
                entry_price=price,
                stop_price=price * (0.99 if side == "long" else 1.01),
                atr=atr,
                rr=None,
                regime=None,
            )
            qty = pos.position_size
            sl_px = pos.stop_loss
            tp_px = pos.take_profit

        elif action_name == "close":
            risk_close = True

        tr = self.executor.execute_order(
            symbol,
            action_name if action_name != "close" else "sell",  # reduce-only for close handled internally
            quantity=qty,
            price=price,
            order_type="market",
            attach_sl=sl_px,
            attach_tp=tp_px,
            risk_close=risk_close,
        )
        return tr

    def _manage_open_positions(self, symbol: str, df5) -> None:
        """
        Tighten stops on favorable moves and optionally take partials in future.
        """
        price = float(df5["close"].iloc[-1])
        atr = TechnicalIndicators.atr(
            df5["high"].tolist(), df5["low"].tolist(), df5["close"].tolist(), period=14
        ) or 0.0

        # Trailing logic is encapsulated in RiskManager.dynamic_stop_management if we tracked positions there.
        # For now, we rely on the ExchangeAPI's shadow SL/TP and our executor is stateless per order.
        # Hook: if you later register_open_position on RiskManager, you can call:
        #   updated = self.risk_manager.dynamic_stop_management(key, price, atr=atr)
        # and then push an update via executor if needed.
        # This scaffold keeps it simple & safe in simulation mode.

        # No-op placeholder (kept for clarity/logging)
        self.logger.debug(f"[PositionCheck] {symbol} price={price:.4f} atr={atr:.6f}")

    # ───────────────────────── State construction ───────────────────────── #

    def _compose_state(self, df5, df15) -> np.ndarray:
        """
        Build a compact feature vector from 5m and 15m frames.
        - price ratio (close/close[-n])
        - normalized SMA short/long
        - RSI
        - 15m trend proxy (SMA cross)
        """
        # 5m
        prices5 = df5["close"].to_numpy()
        norm_price5 = float(prices5[-1] / prices5[max(0, len(prices5) - 50)]) if len(prices5) >= 50 else 1.0

        sma5_short = TechnicalIndicators.sma(df5["close"], window=10)
        sma5_long = TechnicalIndicators.sma(df5["close"], window=50)
        rsi5 = TechnicalIndicators.rsi(df5["close"], window=14)
        s5 = float(sma5_short.iloc[-1]) if sma5_short is not None and not np.isnan(sma5_short.iloc[-1]) else prices5[-1]
        l5 = float(sma5_long.iloc[-1]) if sma5_long is not None and not np.isnan(sma5_long.iloc[-1]) else prices5[-1]
        r5 = float(rsi5.iloc[-1]) if rsi5 is not None and not np.isnan(rsi5.iloc[-1]) else 50.0

        # 15m (setup / regime)
        prices15 = df15["close"].to_numpy()
        sma15_short = TechnicalIndicators.sma(df15["close"], window=20)
        sma15_long = TechnicalIndicators.sma(df15["close"], window=100)
        s15 = float(sma15_short.iloc[-1]) if sma15_short is not None and not np.isnan(sma15_short.iloc[-1]) else prices15[-1]
        l15 = float(sma15_long.iloc[-1]) if sma15_long is not None and not np.isnan(sma15_long.iloc[-1]) else prices15[-1]
        regime15 = 1.0 if s15 > l15 else -1.0

        vol5 = float(df5["volume"].iloc[-1])

        features = np.array([
            norm_price5,
            s5 / prices5[-1],
            l5 / prices5[-1],
            r5 / 100.0,
            vol5,
            s15 / prices15[-1],
            l15 / prices15[-1],
            regime15,
        ], dtype=np.float32)

        if features.size < self.state_size:
            pad = np.zeros(self.state_size - features.size, dtype=np.float32)
            features = np.concatenate([features, pad])
        else:
            features = features[: self.state_size]
        return features

    # ───────────────────────── DQN helpers ───────────────────────── #

    def _select_action(self, state: np.ndarray) -> int:
        if self.training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.policy_net(state_t)
        return int(torch.argmax(q, dim=1).item())

    def _learn(self):
        if len(self.memory) < self.batch_size:
            return
        batch = Experience(*zip(*self.memory.sample(self.batch_size)))

        states = torch.as_tensor(np.stack(batch.state), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(np.stack(batch.next_state), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_curr = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            q_next = self.target_net(next_states).max(1, keepdim=True)[0]
        q_targ = rewards + (self.gamma * q_next * (1 - dones))

        loss = F.mse_loss(q_curr, q_targ)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Soft update
        for t_param, p_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            t_param.data.mul_(1.0 - self.tau).add_(self.tau * p_param.data)

    # ───────────────────────── Misc ───────────────────────── #

    def _action_to_command(self, action_idx: int) -> str:
        mapping = {0: "buy", 1: "sell", 2: "hold", 3: "buy", 4: "sell", 5: "close"}
        return mapping.get(action_idx, "hold")

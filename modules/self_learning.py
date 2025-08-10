# modules/self_learning.py

import random
import time
import numpy as np
from collections import deque, namedtuple
from typing import List, Dict, Optional
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging

from modules.data_manager import DataManager
from modules.error_handler import ErrorHandler
from modules.reward_system import RewardSystem, calculate_points
from modules.trade_executor import TradeExecutor
from modules.technical_indicators import TechnicalIndicators
from modules.risk_management import RiskManager, RiskViolationError
from modules.top_pairs import TopPairs
import config

Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))
    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)


class DQNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        h0, h1, h2 = hidden_dims
        self.fc1 = nn.Linear(input_dim, h0)
        self.fc2 = nn.Linear(h0, h1)
        self.fc3 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, output_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)


class SelfLearningBot:
    """
    DQN-based agent tuned for:
      - 15m setup context, 5m execution
      - Top-pairs rotation (refresh hourly) with spike checks
      - Incremental OHLCV loading via DataManager (no heavy downloads)
      - Simulation-only execution (paper), but obey risk manager sizing
      - Open-position monitoring: once per minute; otherwise 5m cadence
    """

    def __init__(
        self,
        data_provider: Optional[DataManager],
        error_handler: ErrorHandler,
        reward_system: Optional[RewardSystem] = None,
        risk_manager: Optional[RiskManager] = None,
        *,
        state_size: int = 8,            # expanded features
        action_size: int = 6,           # buy/sell/hold/close + extras
        hidden_dims: List[int] = [128, 64, 32],
        batch_size: int = 64,
        gamma: float = 0.99,
        learning_rate: float = 1e-3,
        exploration_max: float = 1.0,
        exploration_min: float = 0.05,
        exploration_decay: float = 0.995,
        memory_size: int = 100_000,
        tau: float = 0.005,
        training: bool = True,
        timeframe_exec: str = "5m",
        timeframe_setup: str = "15m",
        default_symbol: Optional[str] = None,
        top_pairs_refresh_min: int = 60,
        position_watch_interval_sec: int = 60,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(h)
        self.logger.setLevel(getattr(logging, str(getattr(config, "LOG_LEVEL", "INFO")), logging.INFO)
                             if isinstance(getattr(config, "LOG_LEVEL", "INFO"), str) else getattr(config, "LOG_LEVEL", logging.INFO))

        self.error_handler = error_handler
        self.reward_system = reward_system or RewardSystem()
        self.risk_manager = risk_manager

        self.dm = data_provider or DataManager()
        self.top_pairs = TopPairs(data_manager=self.dm, refresh_minutes=top_pairs_refresh_min)

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.tau = tau

        self.epsilon = exploration_max if training else 0.0
        self.epsilon_min = exploration_min
        self.epsilon_decay = exploration_decay
        self.training = training

        # execution in SIMULATION mode only
        self.executor = TradeExecutor(simulation_mode=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = ReplayBuffer(capacity=memory_size)

        self.policy_net = DQNetwork(state_size, hidden_dims, action_size).to(self.device)
        self.target_net = DQNetwork(state_size, hidden_dims, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.train_steps = 0

        self.timeframe_exec = timeframe_exec  # "5m"
        self.timeframe_setup = timeframe_setup  # "15m"
        self.default_symbol = default_symbol or getattr(config, "DEFAULT_SYMBOL", "BTC/USDT")

        self.is_connected = True
        self.is_training = training
        self.is_trading = False
        self.last_heartbeat = time.time()

        # monitoring cadence
        self.position_watch_interval_sec = position_watch_interval_sec
        self._last_pos_check: Dict[str, float] = {}

        # UI metrics
        self.current_balance: float = getattr(config, "SIMULATION_START_BALANCE", 1000.0)
        self.virtual_balance: float = self.current_balance
        self.reward_points: float = 0.0
        self.current_symbol: Optional[str] = self.default_symbol
        self.timeframe: str = self.timeframe_exec

    # ─────────────────────────────
    # Public control hooks
    # ─────────────────────────────
    def start_training(self):
        self.is_training = True
        self.is_trading = False

    def stop_training(self):
        self.is_training = False

    def start_trading(self):
        # still simulation, but toggles UI state
        self.is_trading = True

    def stop_trading(self):
        self.is_trading = False

    # ─────────────────────────────
    # Main loop tick
    # ─────────────────────────────
    def step(self):
        """One loop step: refresh pairs (if needed), iterate symbols."""
        self.last_heartbeat = time.time()
        if self.top_pairs.needs_refresh():
            self.top_pairs.refresh()

        universe = [p["symbol"] for p in (self.top_pairs.current() or [])]
        if not universe:
            # ensure at least default
            universe = [self.default_symbol]

        for sym in universe:
            try:
                self.act_and_learn(sym)
            except Exception as e:
                self.error_handler.handle(e, {"symbol": sym, "stage": "step"})

    def select_action(self, state: np.ndarray) -> int:
        try:
            if self.training and random.random() < self.epsilon:
                return random.randrange(self.action_size)
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return int(torch.argmax(q_values, dim=1).item())
        except Exception as e:
            self.error_handler.log_error(e, {"stage": "select_action"})
            return 2  # hold

    def learn(self):
        try:
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

            for t_param, p_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                t_param.data.mul_(1.0 - self.tau).add_(self.tau * p_param.data)
        except Exception as e:
            self.error_handler.log_error(e, {"stage": "learn"})

    # ─────────────────────────────
    # Act & Learn on a single symbol
    # ─────────────────────────────
    def act_and_learn(self, symbol: Optional[str], timestamp: Optional[datetime] = None):
        sym = symbol or self.default_symbol
        self.current_symbol = sym
        ts = timestamp or datetime.now(timezone.utc)

        # ensure fresh data (incremental; cheap)
        self.dm.update_bars(sym, self.timeframe_exec, bootstrap_candles=600)
        self.dm.update_bars(sym, self.timeframe_setup, bootstrap_candles=600)

        state = self._get_state(sym)
        if state is None:
            # no usable data yet; skip
            return

        # Open-position monitoring: check once a minute if a position exists
        if self._has_open_position(sym):
            now = time.time()
            if now - self._last_pos_check.get(sym, 0) >= self.position_watch_interval_sec:
                self._last_pos_check[sym] = now
                # light monitoring — evaluate close/hold only
                action_idx = self._monitoring_decision(sym, state)
            else:
                # skip frequent re-evaluation; hold
                action_idx = 2  # hold
        else:
            # Normal action selection
            action_idx = self.select_action(state)

        action_name = self._action_to_command(action_idx)

        price = self._safe_last_price(sym, state)
        if price <= 0:
            return

        qty = None
        sl_px, tp_px = None, None
        risk_close = False

        try:
            if action_name in ("buy", "sell"):
                # Execution on 5m, simple ATR stop/take if available
                atr_val = self._latest_atr(sym)
                if action_name == "buy":
                    sl_px = price * (1.0 - 0.01)  # fallback if ATR not available
                    tp_px = price * (1.0 + 0.02)
                    side = "long"
                else:
                    sl_px = price * (1.0 + 0.01)
                    tp_px = price * (1.0 - 0.02)
                    side = "short"

                if self.risk_manager:
                    try:
                        # With ATR-aware SL/TP from risk manager when possible
                        if atr_val:
                            sl_r, tp_r = self.risk_manager.compute_sl_tp_from_atr(
                                "long" if action_name == "buy" else "short",
                                price,
                                atr_val
                            )
                            sl_px, tp_px = sl_r, tp_r

                        pos = self.risk_manager.calculate_position_size(
                            symbol=sym,
                            side="long" if action_name == "buy" else "short",
                            entry_price=price,
                            stop_price=sl_px,
                            atr=atr_val,
                            regime=self._regime_hint(sym),
                        )
                        qty = pos.position_size
                    except RiskViolationError as re:
                        self.logger.info(f"Risk blocked order: {re}")
                        return
                else:
                    # simple fixed-size fallback
                    qty = max(0.0, getattr(config, "MIN_TRADE_AMOUNT_USD", 10.0) / price)

            elif action_name == "close":
                risk_close = True

            tr = self.executor.execute_order(
                sym,
                action_name if action_name != "close" else "sell",
                quantity=qty,
                price=price,
                order_type="market",
                attach_sl=sl_px,
                attach_tp=tp_px,
                risk_close=risk_close,
            )

            entry_price = float(tr.get("entry_price") or price)
            exit_price = float(tr.get("exit_price") or entry_price)
            q_used = float(tr.get("quantity") or (qty or 0.0))

            reward_raw = self.reward_system.calculate_reward(
                entry_price=entry_price,
                exit_price=exit_price,
                position_size=q_used,
                entry_time=ts,
                exit_time=ts,
                max_drawdown=0.0,
                volatility=0.0,
                stop_loss_triggered=False,
            )
            reward_pts = calculate_points(
                profit=reward_raw,
                entry_time=ts,
                exit_time=ts,
                stop_loss_triggered=False,
            )
            reward_total = float(reward_raw + reward_pts)

            next_state = self._get_state(sym) or state
            done = tr.get("status") in ("closed",)

            self.memory.push(state, action_idx, reward_total, next_state, done)
            self.train_steps += 1
            self.reward_points += float(reward_pts)

            # UI feed-through
            bal_now = self.executor.get_balance()
            if bal_now:
                self.current_balance = float(bal_now)
                self.virtual_balance = float(bal_now)

            self.logger.info(
                f"Step {self.train_steps} {sym} | action={action_name} qty={q_used:.8f} "
                f"reward={reward_total:.4f} eps={self.epsilon:.4f}"
            )

            if self.training:
                self.learn()
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        except Exception as e:
            self.error_handler.handle(e, {"symbol": sym})

    # ─────────────────────────────
    # State construction
    # ─────────────────────────────
    def _get_state(self, symbol: str) -> Optional[np.ndarray]:
        try:
            df_exec = self.dm.load_historical_data(symbol, self.timeframe_exec, auto_update=False)
            df_setup = self.dm.load_historical_data(symbol, self.timeframe_setup, auto_update=False)
            if len(df_exec) < 25 or len(df_setup) < 25:
                return None
        except Exception as e:
            self.error_handler.handle(e, {"symbol": symbol, "stage": "load_state"})
            return None

        # features from 5m (current price context)
        tail_e = df_exec.iloc[-50:].copy()
        prices_e = tail_e["close"].to_numpy(dtype=float)
        norm_price = float(prices_e[-1] / prices_e[0]) if prices_e[0] else 0.0

        sma_s_e = TechnicalIndicators.sma(tail_e["close"], window=10)
        sma_l_e = TechnicalIndicators.sma(tail_e["close"], window=50)
        rsi_e = TechnicalIndicators.rsi(tail_e["close"], window=14)

        s_short_e = float(sma_s_e.iloc[-1]) if sma_s_e is not None and not np.isnan(sma_s_e.iloc[-1]) else 0.0
        s_long_e = float(sma_l_e.iloc[-1]) if sma_l_e is not None and not np.isnan(sma_l_e.iloc[-1]) else 0.0
        rsi_last_e = float(rsi_e.iloc[-1]) if rsi_e is not None and not np.isnan(rsi_e.iloc[-1]) else 0.0

        # features from 15m (setup/regime)
        tail_s = df_setup.iloc[-100:].copy()
        sma_s_s = TechnicalIndicators.sma(tail_s["close"], window=10)
        sma_l_s = TechnicalIndicators.sma(tail_s["close"], window=50)
        rsi_s = TechnicalIndicators.rsi(tail_s["close"], window=14)

        s_short_s = float(sma_s_s.iloc[-1]) if sma_s_s is not None and not np.isnan(sma_s_s.iloc[-1]) else 0.0
        s_long_s = float(sma_l_s.iloc[-1]) if sma_l_s is not None and not np.isnan(sma_l_s.iloc[-1]) else 0.0
        rsi_last_s = float(rsi_s.iloc[-1]) if rsi_s is not None and not np.isnan(rsi_s.iloc[-1]) else 0.0

        last_close = float(prices_e[-1])
        features = [
            norm_price,
            (s_short_e / last_close) if last_close else 0.0,
            (s_long_e / last_close) if last_close else 0.0,
            rsi_last_e / 100.0,
            (s_short_s / (float(tail_s["close"].iloc[-1]) or 1.0)) if len(tail_s) else 0.0,
            (s_long_s / (float(tail_s["close"].iloc[-1]) or 1.0)) if len(tail_s) else 0.0,
            rsi_last_s / 100.0,
            float(tail_e["volume"].iloc[-1]),
        ]
        state = np.array(features, dtype=np.float32)
        return state

    # ─────────────────────────────
    # Helpers
    # ─────────────────────────────
    def _action_to_command(self, action_idx: int) -> str:
        mapping = {0: "buy", 1: "sell", 2: "hold", 3: "buy", 4: "sell", 5: "close"}
        return mapping.get(action_idx, "hold")

    def _safe_last_price(self, symbol: str, state: np.ndarray) -> float:
        try:
            df = self.dm.load_historical_data(symbol, timeframe=self.timeframe_exec, auto_update=False)
            return float(df["close"].iloc[-1])
        except Exception:
            return float(state[0]) if state is not None and state.size > 0 else 0.0

    def _latest_atr(self, symbol: str) -> Optional[float]:
        try:
            df = self.dm.load_historical_data(symbol, timeframe=self.timeframe_exec, auto_update=False)
            if len(df) < 20:
                return None
            h = df["high"].tolist()
            l = df["low"].tolist()
            c = df["close"].tolist()
            from modules.technical_indicators import atr
            return float(atr(h, l, c, period=14) or 0.0) or None
        except Exception:
            return None

    def _regime_hint(self, symbol: str) -> Optional[str]:
        """
        Simple regime hint from 15m SMA cross.
        """
        try:
            df = self.dm.load_historical_data(symbol, timeframe=self.timeframe_setup, auto_update=False)
            if len(df) < 50:
                return None
            sma_s = TechnicalIndicators.sma(df["close"], window=20)
            sma_l = TechnicalIndicators.sma(df["close"], window=50)
            if sma_s is None or sma_l is None:
                return None
            vs = float(sma_s.iloc[-1])
            vl = float(sma_l.iloc[-1])
            return "trend" if vs > vl else "range"
        except Exception:
            return None

    def _has_open_position(self, symbol: str) -> bool:
        try:
            sym = self.executor.exchange._resolve_symbol(symbol)
            pos = getattr(self.executor.exchange, "_sim_positions", {}).get(sym)
            if pos and float(pos.get("quantity", 0.0)) > 0:
                return True
            return False
        except Exception:
            return False

    def _monitoring_decision(self, symbol: str, state: np.ndarray) -> int:
        """
        Lightweight monitoring when a position exists: bias to hold/close.
        Very conservative: if RSI diverges strongly against the trade, suggest close.
        """
        # decode minimal info from state (rsi_e at index 3)
        rsi_e = float(state[3] * 100.0)
        try:
            sym = self.executor.exchange._resolve_symbol(symbol)
            pos = getattr(self.executor.exchange, "_sim_positions", {}).get(sym)
            if not pos:
                return 2  # hold
            side = pos.get("side", "long")
            # naive guardrails
            if side == "long" and rsi_e > 75:
                return 5  # close
            if side == "short" and rsi_e < 25:
                return 5  # close
        except Exception:
            pass
        return 2  # hold

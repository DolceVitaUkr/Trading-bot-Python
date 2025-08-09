# modules/self_learning.py
import random
import numpy as np
from collections import deque, namedtuple
from typing import List, Dict, Optional
from datetime import datetime, timezone
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from modules.data_manager import DataManager
from modules.error_handler import ErrorHandler
from modules.reward_system import RewardSystem
from modules.trade_executor import TradeExecutor
from modules.technical_indicators import TechnicalIndicators
from modules.risk_management import RiskManager, RiskViolationError

try:
    import pandas as pd
except Exception:  # very unlikely, but guard NaN checks
    pd = None  # type: ignore

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
        h1, h2, h3 = hidden_dims
        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, output_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)


class SelfLearningBot:
    """
    DQN-based agent with minimal defaults to avoid breaking older wiring.
    """

    def __init__(
        self,
        data_provider: DataManager,
        error_handler: ErrorHandler,
        reward_system: Optional[RewardSystem] = None,
        risk_manager: Optional[RiskManager] = None,
        state_size: int = 5,
        action_size: int = 6,  # buy/sell/hold/close + extras
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
        timeframe: str = "15m",
        symbol: Optional[str] = None,
    ):
        self.data_provider = data_provider
        self.error_handler = error_handler
        self.reward_system = reward_system or RewardSystem()
        self.risk_manager = risk_manager

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

        self.executor = TradeExecutor(simulation_mode=training)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = ReplayBuffer(capacity=memory_size)

        self.policy_net = DQNetwork(state_size, hidden_dims, action_size).to(self.device)
        self.target_net = DQNetwork(state_size, hidden_dims, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.train_steps = 0
        self.logger = logging.getLogger(self.__class__.__name__)

        self.timeframe = timeframe
        self.default_symbol = symbol

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

            # Soft update target net
            for t_param, p_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                t_param.data.mul_(1.0 - self.tau).add_(self.tau * p_param.data)
        except Exception as e:
            self.error_handler.log_error(e, {"stage": "learn"})

    def act_and_learn(self, symbol: Optional[str], timestamp: Optional[datetime] = None):
        sym = symbol or self.default_symbol
        if not sym:
            self.logger.warning("act_and_learn called without symbol")
            return

        ts = timestamp or datetime.now(timezone.utc)

        try:
            state = self._get_state(sym)
            action_idx = self.select_action(state)
            action_name = self._action_to_command(action_idx)

            price = self._safe_last_price(sym, state)

            qty = None
            sl_px, tp_px = None, None
            risk_close = False

            if action_name in ("buy", "sell"):
                is_buy = action_name == "buy"
                side_str = "long" if is_buy else "short"

                # initial SL/TP (will be refined by RiskManager if wired)
                sl_mult = 0.99 if is_buy else 1.01
                sl_px = price * sl_mult

                if self.risk_manager:
                    try:
                        pos = self.risk_manager.calculate_position_size(
                            symbol=sym,
                            side=side_str,
                            entry_price=price,
                            stop_price=sl_px,
                            atr=None,
                            rr=None,
                            regime=None,
                        )
                        qty = float(pos.position_size)
                        sl_px = float(pos.stop_loss)
                        tp_px = float(pos.take_profit)
                    except RiskViolationError as re:
                        self.logger.info(f"Risk blocked order: {re}")
                        return
                else:
                    # naive qty: 5% notional
                    notional = price * 0.05
                    qty = max(1e-6, notional / max(price, 1e-9))
                    tp_px = price * (1.02 if is_buy else 0.98)

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

            reward_score = self.reward_system.calculate_reward(
                entry_price=entry_price,
                exit_price=exit_price,
                position_size=q_used,
                entry_time=ts,
                exit_time=ts,
                max_drawdown=0.0,
                volatility=0.0,
                stop_loss_triggered=False,
            )

            next_state = self._get_state(sym)
            done = tr.get("status") in ("closed",)

            self.memory.push(state, action_idx, float(reward_score), next_state, done)
            self.train_steps += 1
            self.logger.info(
                f"Step {self.train_steps} {sym} | action={action_name} qty={q_used:.8f} "
                f"reward={float(reward_score):.4f} eps={self.epsilon:.4f}"
            )

            if self.training:
                self.learn()
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        except Exception as e:
            self.error_handler.handle(e, {"symbol": sym})

    def _get_state(self, symbol: str) -> np.ndarray:
        try:
            df = self.data_provider.load_historical_data(symbol, timeframe=self.timeframe)
            if df.empty:
                return np.zeros(self.state_size, dtype=np.float32)
        except Exception as e:
            self.error_handler.handle(e, {"symbol": symbol, "stage": "load_state"})
            return np.zeros(self.state_size, dtype=np.float32)

        lookback = min(len(df), 50)
        tail = df.iloc[-lookback:].copy()

        prices = tail["close"].to_numpy()
        base = prices[0] if len(prices) else 0.0
        norm_price = float(prices[-1] / base) if base else 0.0

        tail["sma_short"] = TechnicalIndicators.sma(tail["close"], window=10)
        tail["sma_long"] = TechnicalIndicators.sma(tail["close"], window=50)
        tail["rsi"] = TechnicalIndicators.rsi(tail["close"], window=14)
        last = tail.iloc[-1]

        def safe_val(x) -> float:
            if pd is not None and hasattr(pd, "isna") and pd.isna(x):
                return 0.0
            return float(x) if x is not None else 0.0

        sma_short = safe_val(last.get("sma_short"))
        sma_long = safe_val(last.get("sma_long"))
        rsi_last = safe_val(last.get("rsi"))
        vol_last = safe_val(last.get("volume"))
        px_last = prices[-1] if len(prices) else 0.0

        features = [
            norm_price,
            (sma_short / px_last) if px_last else 0.0,
            (sma_long / px_last) if px_last else 0.0,
            rsi_last / 100.0,
            vol_last,
        ]
        state = np.array(features, dtype=np.float32)

        if state.size < self.state_size:
            pad = np.zeros(self.state_size - state.size, dtype=np.float32)
            state = np.concatenate([state, pad])
        else:
            state = state[: self.state_size]
        return state

    def _action_to_command(self, action_idx: int) -> str:
        # 0: buy, 1: sell, 2: hold, 3: buy (add), 4: sell (add), 5: close
        mapping = {0: "buy", 1: "sell", 2: "hold", 3: "buy", 4: "sell", 5: "close"}
        return mapping.get(action_idx, "hold")

    def _safe_last_price(self, symbol: str, state: np.ndarray) -> float:
        try:
            df = self.data_provider.load_historical_data(symbol, timeframe=self.timeframe)
            if not df.empty:
                return float(df["close"].iloc[-1])
        except Exception:
            pass
        # fallback: 1.0 (neutral) if state has normalized price feature
        return 1.0

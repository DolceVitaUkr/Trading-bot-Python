#modules/self_learning.py

import os
import random
import numpy as np
from collections import deque, namedtuple
from typing import List, Dict, Optional
from datetime import datetime
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

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

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
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.out = nn.Linear(hidden_dims[2], output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)

class SelfLearningBot:
    def __init__(
        self,
        data_provider: DataManager,
        error_handler: ErrorHandler,
        reward_system: RewardSystem,
        risk_manager: Optional[RiskManager],
        state_size: int,
        action_size: int = 10,
        hidden_dims: List[int] = [128, 64, 32],
        batch_size: int = 64,
        gamma: float = 0.99,
        learning_rate: float = 1e-3,
        exploration_max: float = 1.0,
        exploration_min: float = 0.1,
        exploration_decay: float = 0.995,
        memory_size: int = 100000,
        tau: float = 0.001,
        training: bool = True
    ):
        self.data_provider = data_provider
        self.error_handler = error_handler
        self.reward_system = reward_system
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

    def select_action(self, state: np.ndarray) -> int:
        try:
            if self.training and random.random() < self.epsilon:
                return random.randrange(self.action_size)
            state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return torch.argmax(q_values, dim=1).item()
        except Exception as e:
            self.error_handler.log_error(e, {'state': state.tolist()})
            return 2  # hold

    def learn(self):
        try:
            if len(self.memory) < self.batch_size:
                return
            batch = Experience(*zip(*self.memory.sample(self.batch_size)))
            states = torch.FloatTensor(np.stack(batch.state)).to(self.device)
            actions = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
            rewards = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(np.stack(batch.next_state)).to(self.device)
            dones = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)

            current_q = self.policy_net(states).gather(1, actions)
            with torch.no_grad():
                next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (self.gamma * next_q * (1 - dones))

            loss = F.mse_loss(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()

            for t_param, p_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                t_param.data.copy_(self.tau * p_param.data + (1.0 - self.tau) * t_param.data)
        except Exception as e:
            self.error_handler.log_error(e, {'stage': 'learn'})

    def act_and_learn(self, symbol: str, timestamp: datetime):
        try:
            state = self._get_state(symbol)
            action_idx = self.select_action(state)
            action_name = self._action_to_command(action_idx)

            price = self._safe_get_price(symbol, state)
            qty = 1.0

            if self.risk_manager and action_name in ['buy', 'sell']:
                try:
                    stop_price = price * (0.99 if action_name == 'buy' else 1.01)
                    qty = self.risk_manager.calculate_position_size(price, stop_price)
                except RiskViolationError as re:
                    self.logger.warning(f"Risk limit hit: {re}")
                    return

            trade_result = self.executor.execute_order(symbol, action_name, qty, price)

            reward_raw = self.reward_system.calculate_reward(
                trade_result.get('entry_price', price),
                trade_result.get('exit_price', price),
                trade_result.get('quantity', qty),
                trade_result.get('entry_time', timestamp),
                trade_result.get('exit_time', timestamp),
                max_drawdown=trade_result.get('max_drawdown', 0.0),
                volatility=trade_result.get('volatility', 0.0),
                stop_loss_triggered=trade_result.get('stop_loss_triggered', False)
            )

            reward_points = calculate_points(
                profit=reward_raw,
                entry_time=trade_result.get('entry_time', timestamp),
                exit_time=trade_result.get('exit_time', timestamp),
                stop_loss_triggered=trade_result.get('stop_loss_triggered', False)
            )

            reward_total = reward_raw + reward_points
            next_state = self._get_state(symbol)
            done = trade_result.get('done', False)

            self.memory.push(state, action_idx, reward_total, next_state, done)
            self.train_steps += 1
            self.logger.info(f"Step {self.train_steps}: {action_name} -> Reward {reward_total:.4f}")

            if self.training:
                self.learn()
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        except Exception as e:
            self.error_handler.handle(e, {'symbol': symbol})

    def _get_state(self, symbol: str) -> np.ndarray:
        try:
            df = self.data_provider.load_historical_data(symbol, timeframe='1h')
        except Exception as e:
            self.error_handler.handle(e, {'symbol': symbol})
            return np.zeros(self.state_size, dtype=np.float32)

        if df.empty:
            return np.zeros(self.state_size, dtype=np.float32)

        prices = df['close'].values[-50:]
        norm_price = prices[-1] / prices[0] if prices[0] != 0 else 0.0

        df['sma_short'] = TechnicalIndicators.sma(df['close'], window=10)
        df['sma_long'] = TechnicalIndicators.sma(df['close'], window=50)
        df['rsi'] = TechnicalIndicators.rsi(df['close'], window=14)
        latest = df.iloc[-1]

        features = [
            norm_price,
            latest['sma_short'] / prices[-1] if latest['sma_short'] and prices[-1] else 0.0,
            latest['sma_long'] / prices[-1] if latest['sma_long'] and prices[-1] else 0.0,
            (latest['rsi'] or 0) / 100.0,
            float(latest['volume'])
        ]

        state = np.array(features, dtype=np.float32)
        return np.pad(state, (0, max(0, self.state_size - len(state))))[:self.state_size]

    def _action_to_command(self, action_idx: int) -> str:
        action_map = {
            0: 'buy', 1: 'sell', 2: 'hold',
            3: 'buy', 4: 'sell', 5: 'close',
            6: 'buy', 7: 'sell', 8: 'buy', 9: 'sell'
        }
        return action_map.get(action_idx, 'hold')

    def _safe_get_price(self, symbol: str, state: np.ndarray) -> float:
        try:
            df = self.data_provider.load_historical_data(symbol, timeframe='1h')
            return df['close'].iloc[-1]
        except:
            return state[0] if state.size > 0 else 0.0

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
from modules.reward_system import RewardSystem
from modules.trade_executor import TradeExecutor
from modules.technical_indicators import TechnicalIndicators

# Experience tuple for replay memory
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples for experience replay.
    """
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        e = Experience(state, action, reward, next_state, done)
        self.buffer.append(e)
    
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)

class DQNetwork(nn.Module):
    """
    Deep Q-Network with 3 hidden layers.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super(DQNetwork, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.out = nn.Linear(hidden_dims[2], output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)

class SelfLearningBot:
    """
    Deep Q-Learning agent for trading.
    """
    def __init__(
        self,
        data_provider: DataManager,
        error_handler: ErrorHandler,
        reward_system: RewardSystem,
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
        # Set up dependencies
        self.data_provider = data_provider
        self.error_handler = error_handler
        self.reward_system = reward_system
        
        # RL hyperparameters
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
        
        # Trading components
        self.executor = TradeExecutor(simulation_mode=training)
        
        # Device configuration (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize replay memory
        self.memory = ReplayBuffer(capacity=memory_size)
        
        # Build Q-networks (policy and target)
        self.policy_net = DQNetwork(state_size, hidden_dims, action_size).to(self.device)
        self.target_net = DQNetwork(state_size, hidden_dims, action_size).to(self.device)
        # Initialize target network parameters to match policy network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # For logging/tracking
        self.train_steps = 0
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Select an action using epsilon-greedy policy from given state.
        """
        try:
            state_tensor = torch.FloatTensor(state).to(self.device)
            if self.training and random.random() < self.epsilon:
                action = random.randrange(self.action_size)
            else:
                self.policy_net.eval()
                with torch.no_grad():
                    q_values = self.policy_net(state_tensor.unsqueeze(0))
                    action = torch.argmax(q_values, dim=1).item()
            return action
        except Exception as e:
            # On error, log and return default action
            self.error_handler.log_error(e, {'state': state.tolist() if isinstance(state, np.ndarray) else state})
            return 2 if self.action_size > 2 else 0
    
    def learn(self):
        """
        Sample a batch from memory and perform a learning step.
        """
        try:
            if len(self.memory) < self.batch_size:
                return
            
            experiences = self.memory.sample(self.batch_size)
            batch = Experience(*zip(*experiences))
            
            states = torch.FloatTensor(np.stack(batch.state)).to(self.device)
            actions = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
            rewards = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(np.stack(batch.next_state)).to(self.device)
            dones = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)
            
            # Compute current Q values
            current_q = self.policy_net(states).gather(1, actions)
            # Compute next Q values from target network
            with torch.no_grad():
                next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            # Compute target Q values
            target_q = rewards + (self.gamma * next_q * (1 - dones))
            
            loss = F.mse_loss(current_q, target_q)
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Soft update of target network parameters
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
        except Exception as e:
            self.error_handler.log_error(e, {'stage': 'learn'})
    
    def act_and_learn(self, symbol: str, timestamp: datetime):
        """
        Construct state, select action, execute trade, store experience, and learn.
        """
        try:
            state = self._get_state(symbol)
            action_idx = self.select_action(state)
            action_name = self._action_to_command(action_idx)
            
            # Get latest price for execution
            try:
                df = self.data_provider.load_historical_data(symbol, timeframe='1h')
                price = df['close'].iloc[-1]
            except Exception:
                price = state[0] if state is not None else None
            
            quantity = 1.0  # fixed quantity for simplicity
            
            # Execute trade
            try:
                trade_result = self.executor.execute_order(
                    symbol=symbol,
                    side=action_name,
                    quantity=quantity,
                    price=price
                )
            except Exception as e:
                self.error_handler.handle(e, {'symbol': symbol, 'action_idx': action_idx, 'action_name': action_name})
                return
            
            # Calculate reward
            entry_price = trade_result.get('entry_price', trade_result.get('price', 0.0))
            exit_price = trade_result.get('exit_price', trade_result.get('price', 0.0))
            position_size = trade_result.get('quantity', trade_result.get('amount', 0.0))
            entry_time = trade_result.get('entry_time', timestamp)
            exit_time = trade_result.get('exit_time', timestamp)
            reward = self.reward_system.calculate_reward(
                entry_price=entry_price,
                exit_price=exit_price,
                position_size=position_size,
                entry_time=entry_time,
                exit_time=exit_time,
                max_drawdown=trade_result.get('max_drawdown', 0.0),
                volatility=trade_result.get('volatility', 0.0),
                stop_loss_triggered=trade_result.get('stop_loss_triggered', False)
            )
            
            next_state = self._get_state(symbol)
            done = trade_result.get('done', False)
            
            self.memory.push(state, action_idx, reward, next_state, done)
            
            self.train_steps += 1
            self.logger.info(f"Step {self.train_steps}: Action {action_idx} ({action_name}), Reward {reward:.4f}, Epsilon {self.epsilon:.4f}")
            
            if self.training:
                self.learn()
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        except Exception as e:
            self.error_handler.handle(e, {'symbol': symbol})
    
    def save_model(self, path: str):
        """
        Save model checkpoint to given path.
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({
                'policy_state_dict': self.policy_net.state_dict(),
                'target_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'train_steps': self.train_steps
            }, path)
            self.logger.info(f"Model saved to {path}")
        except Exception as e:
            self.error_handler.handle(e, {'path': path})
    
    def load_model(self, path: str):
        """
        Load model checkpoint from given path.
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.train_steps = checkpoint.get('train_steps', self.train_steps)
            self.logger.info(f"Model loaded from {path}")
        except Exception as e:
            self.error_handler.handle(e, {'path': path})
    
    def _get_state(self, symbol: str) -> np.ndarray:
        """
        Construct the state representation from historical market data.
        Includes recent OHLCV values and technical indicators.
        """
        try:
            df = self.data_provider.load_historical_data(symbol, timeframe='1h')
        except Exception as e:
            self.error_handler.handle(e, {'symbol': symbol})
            return np.zeros(self.state_size, dtype=np.float32)
        
        lookback = min(len(df), max(1, self.state_size // 2))
        df_tail = df.iloc[-lookback:].copy()
        prices = df_tail['close'].values
        if len(prices) == 0:
            return np.zeros(self.state_size, dtype=np.float32)
        # Normalize prices
        norm_price = prices[-1] / prices[0] if prices[0] != 0 else 0.0
        
        # Compute technical indicators
        df_tail['sma_short'] = TechnicalIndicators.sma(df_tail['close'], window=10)
        df_tail['sma_long'] = TechnicalIndicators.sma(df_tail['close'], window=50)
        df_tail['rsi'] = TechnicalIndicators.rsi(df_tail['close'], window=14)
        latest = df_tail.iloc[-1]
        
        # Feature vector
        features = [
            norm_price,
            (latest['sma_short'] / prices[-1]) if latest['sma_short'] is not None and prices[-1] != 0 else 0.0,
            (latest['sma_long'] / prices[-1]) if latest['sma_long'] is not None and prices[-1] != 0 else 0.0,
            (latest['rsi'] / 100.0) if latest['rsi'] is not None else 0.0,
            float(latest['volume'])
        ]
        state = np.array(features, dtype=np.float32)
        if state.size < self.state_size:
            pad_size = self.state_size - state.size
            state = np.concatenate([state, np.zeros(pad_size, dtype=np.float32)])
        else:
            state = state[:self.state_size]
        return state
    
    def _action_to_command(self, action_idx: int) -> str:
        """
        Convert discrete action index to trade command (side).
        """
        action_map = {
            0: 'buy',
            1: 'sell',
            2: 'hold',
            3: 'buy',      # treat 'increase' as buy
            4: 'sell',     # treat 'decrease' as sell
            5: 'hold',     # treat 'close' as hold or close
            6: 'buy',
            7: 'sell',
            8: 'buy',
            9: 'sell'
        }
        return action_map.get(action_idx, 'hold')

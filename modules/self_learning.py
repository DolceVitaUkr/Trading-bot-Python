# modules/self_learning.py
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import deque, namedtuple
import random
from typing import List, Dict
import config
from utils.utilities import format_timestamp


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define experience tuple
Experience = namedtuple('Experience', 
    ['state', 'action', 'reward', 'next_state', 'done'])

class TradingDataset(Dataset):
    """PyTorch Dataset for experience replay"""
    def __init__(self, experiences):
        self.experiences = experiences
        
    def __len__(self):
        return len(self.experiences)
    
    def __getitem__(self, idx):
        experience = self.experiences[idx]
        return (torch.FloatTensor(experience.state),
                torch.LongTensor([experience.action]),
                torch.FloatTensor([experience.reward]),
                torch.FloatTensor(experience.next_state),
                torch.FloatTensor([experience.done]))

class DRQN(nn.Module):
    """Deep Recurrent Q-Network with attention mechanism"""
    def __init__(self, input_size, hidden_size, output_size):
        super(DRQN, self).__init__()
        self.hidden_size = hidden_size
        
        # Feature extraction
        self.feature = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )
        
        # Temporal processing
        self.lstm = nn.LSTM(256, hidden_size, batch_first=True)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4)
        
        # Decision making
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Feature extraction
        x = self.feature(x)
        
        # LSTM processing
        x, hidden = self.lstm(x, hidden)
        
        # Attention
        x, _ = self.attention(x, x, x)
        
        # Final decision
        x = self.fc(x[:, -1, :])  # Use last sequence element
        return x, hidden

class SelfLearningBot:
    def __init__(self, state_size=25, action_size=3):
        # Hyperparameters
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = 64
        self.memory_size = 100000
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.tau = 0.001  # For soft target update
        self.sequence_length = 10  # Time steps for sequence learning
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.policy_net = DRQN(state_size, 128, action_size).to(self.device)
        self.target_net = DRQN(state_size, 128, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=3e-4, weight_decay=1e-5)
        
        # Memory
        self.memory = deque(maxlen=self.memory_size)
        self.current_sequence = []
        
        # Training state
        self.hidden_state = None
        self.train_iterations = 0

    def _get_state_features(self, market_state: Dict) -> List[float]:
        """Create sophisticated state representation"""
        return [
            # Price features
            market_state['price'],
            market_state['ema_50'],
            market_state['ema_200'],
            market_state['rsi'],
            market_state['obv'],
            market_state['atr'],
            
            # Volume features
            market_state['volume'],
            market_state['volume_ma_20'],
            
            # Volatility features
            market_state['volatility_30m'],
            market_state['volatility_4h'],
            
            # Market context
            market_state['market_sentiment'],
            market_state['spread'],
            
            # Portfolio state
            market_state['wallet_balance'],
            market_state['position_size'],
            market_state['risk_exposure'],
            
            # Macro features
            market_state['btc_dominance'],
            market_state['fear_greed_index'],
            
            # Temporal features
            market_state['hour_of_day'],
            market_state['day_of_week'],
            
            # Risk management
            market_state['max_drawdown'],
            market_state['sharpe_ratio'],
            
            # Order book features
            market_state['bid_ask_ratio'],
            market_state['order_book_depth'],
            
            # Derivatives market
            market_state['funding_rate'],
            market_state['open_interest']
        ]

    def _calculate_reward(self, trade_result: Dict) -> float:
        """Sophisticated reward function"""
        reward = 0
        
        # Profit-based component
        if trade_result['profit'] > 0:
            # Logarithmic scaling for profits
            reward += np.log1p(trade_result['profit']) * 10
        else:
            # Exponential penalty for losses
            reward -= np.exp(abs(trade_result['profit'])) * 5
            
        # Risk-adjusted return component
        sharpe_contribution = trade_result['sharpe_ratio'] * 2
        reward += sharpe_contribution
        
        # Time penalty
        reward -= trade_result['holding_period'] * 0.1
        
        # Drawdown penalty
        reward -= trade_result['max_drawdown'] * 100
        
        # Volatility adjustment
        reward *= 1 + (trade_result['volatility'] / 100)
        
        # Stop loss penalty
        if trade_result['stop_loss_triggered']:
            reward -= 50
            
        return reward

    def store_experience(self, state: Dict, action: int, 
                       next_state: Dict, reward: float, done: bool):
        """Store experiences in sequential format"""
        processed_state = self._get_state_features(state)
        processed_next_state = self._get_state_features(next_state)
        
        self.current_sequence.append(
            (processed_state, action, reward, processed_next_state, done)
        )
        
        if len(self.current_sequence) >= self.sequence_length or done:
            # Convert sequence to tensor and store
            self.memory.append(list(self.current_sequence))
            self.current_sequence = []

    def _get_batch(self) -> List[List[Experience]]:
        """Sample randomized sequence batches"""
        batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        return batch

    def train(self, simulation_data: List[Dict]):
        """Train the model using experience replay"""
        if len(self.memory) < self.batch_size:
            return

        # Sample batch of sequences
        batch = self._get_batch()
        
        # Convert to tensors
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for sequence in batch:
            seq_states = []
            seq_actions = []
            seq_rewards = []
            seq_next_states = []
            seq_dones = []
            
            for exp in sequence:
                seq_states.append(exp[0])
                seq_actions.append(exp[1])
                seq_rewards.append(exp[2])
                seq_next_states.append(exp[3])
                seq_dones.append(exp[4])
                
            states.append(seq_states)
            actions.append(seq_actions)
            rewards.append(seq_rewards)
            next_states.append(seq_next_states)
            dones.append(seq_dones)
        
        # Training step
        self._train_step(states, actions, rewards, next_states, dones)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update target network
        self._soft_update_target_network()

    def _train_step(self, states, actions, rewards, next_states, dones):
        """Perform a single training iteration"""
        self.policy_net.train()
        self.target_net.eval()
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Get current Q values
        q_values, _ = self.policy_net(states)
        current_q = q_values.gather(2, actions.unsqueeze(2)).squeeze(2)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values, _ = self.target_net(next_states)
            max_next_q = next_q_values.max(2)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Compute loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.train_iterations += 1
        logger.info(f"Training Iteration: {self.train_iterations}, Loss: {loss.item():.4f}")

    def _soft_update_target_network(self):
        """Soft update target network parameters"""
        for target_param, policy_param in zip(self.target_net.parameters(), 
                                            self.policy_net.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1 - self.tau) * target_param.data
            )

    def predict(self, state: Dict) -> str:
        """Make trading decision with exploration and improved state handling"""
        try:
            # Convert state dictionary to properly ordered tensor
            processed_state = self._get_state_features(state)
            
            # Ensure numerical stability and proper tensor shape
            processed_state = np.array(processed_state, dtype=np.float32)
            if np.any(np.isnan(processed_state)):
                processed_state = np.nan_to_num(processed_state)
                
            # Create batch and sequence dimensions (for RNN models)
            state_tensor = torch.FloatTensor(processed_state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
            state_tensor = state_tensor.unsqueeze(1)  # Add sequence dimension if needed

            # Exploration-exploitation tradeoff
            if self.training and np.random.rand() <= self.epsilon:
                return random.choice(["buy", "sell", "hold"])

            # Model prediction with hidden state management
            self.policy_net.eval()
            with torch.no_grad():
                if self.hidden_state is None or self.hidden_state.size(1) != state_tensor.size(0):
                    # Reinitialize hidden state for new batch
                    self.hidden_state = self.policy_net.init_hidden(1).to(self.device)
                
                q_values, self.hidden_state = self.policy_net(
                    state_tensor, self.hidden_state
                )
                
                # Detach hidden state to prevent backprop through time
                self.hidden_state = self.hidden_state.detach()

                # Apply temperature to action probabilities
                action_probs = F.softmax(q_values / self.temperature, dim=-1)
                action = action_probs.argmax().item()

            return ["buy", "sell", "hold"][action]
        
        except Exception as e:
            self.error_handler.log_error(e)
            return "hold"  # Fallback to hold position on error

    def save_model(self, path: str):
        """Save model state"""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_iterations': self.train_iterations
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model state"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.train_iterations = checkpoint['train_iterations']
        logger.info(f"Model loaded from {path}")

# Example usage
if __name__ == "__main__":
    bot = SelfLearningBot()
    
    # Example training loop
    for epoch in range(10):
        # Generate mock experiences
        experiences = [Experience(
            state=np.random.randn(25).tolist(),
            action=np.random.randint(0, 3),
            reward=np.random.randn(),
            next_state=np.random.randn(25).tolist(),
            done=False
        ) for _ in range(100)]
        
        bot.train(experiences)
    
    # Save model
    bot.save_model("trading_model.pth")
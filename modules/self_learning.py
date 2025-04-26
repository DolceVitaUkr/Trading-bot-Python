# modules/self_learning.py
import logging, random
from collections import deque, namedtuple
from datetime import datetime
from typing import Dict
from modules.trade_executor import TradeExecutor
from modules.data_manager import DataManager
from modules.reward_system import RewardSystem

logger = logging.getLogger(__name__)
Experience = namedtuple("Experience", ["state","action","reward","next_state","done"])

class SelfLearningBot:
    def __init__(self,
                 state_size: int=25,
                 action_size: int=3,
                 exploration_multiplier: float=1.0,
                 training: bool=True):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0 if training else 0.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.1
        self.training = training

        self.executor = TradeExecutor(simulation_mode=training)
        self.data_manager = DataManager(test_mode=training)
        self.reward_sys = RewardSystem()
        self.memory = deque(maxlen=100000)

    def act_and_learn(self, state: Dict, timestamp: datetime):
        # Main action
        if self.training and random.random() < self.epsilon:
            main_act = random.randrange(self.action_size)
        else:
            main_act = self._greedy_action(state)
        res = self.executor.execute_order(
            symbol=state["symbol"],
            side=self._action_to_side(main_act),
            amount=state["position_size"],
            price=state["price"]
        )
        reward = self.reward_sys.calculate_reward(
            entry_price=res["entry_price"],
            exit_price=res["exit_price"],
            position_size=res["quantity"],
            entry_time=res["entry_time"],
            exit_time=res["exit_time"],
            max_drawdown=res.get("max_drawdown",0.0),
            volatility=res.get("volatility",0.0),
            stop_loss_triggered=res.get("stop_loss_triggered",False)
        )
        # Store only main trade
        self.memory.append(Experience(state,main_act,reward,state, res.get("done",False)))
        # Exploratory
        for a in range(self.action_size):
            if a==main_act: continue
            try:
                self.executor.simulation_mode = True
                self.executor.execute_order(
                    symbol=state["symbol"],
                    side=self._action_to_side(a),
                    amount=state["position_size"],
                    price=state["price"]
                )
            except:
                pass
        # decay
        if self.training:
            self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)

    def _greedy_action(self, state: Dict) -> int:
        return random.randrange(self.action_size)

    def _action_to_side(self, idx: int) -> str:
        return ["buy","sell","hold"][idx]

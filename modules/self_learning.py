# modules/self_learning.py

import random
import numpy as np
from collections import deque, namedtuple
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone, timedelta
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import config
from modules.data_manager import DataManager
from modules.error_handler import ErrorHandler
from modules.reward_system import RewardSystem, calculate_points
from modules.trade_executor import TradeExecutor
from modules.technical_indicators import TechnicalIndicators
from modules.risk_management import RiskManager, RiskViolationError
from modules.top_pairs import TopPairs

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
    """
    DQN-based agent aligned to:
      - Simulation-only execution
      - 5m & 15m intervals (15m = setup scan, 5m = entries/management)
      - Live market data feed (WS) + REST backfill with incremental append
      - Hourly Top Pairs refresh; minute-level open-position monitoring
      - No re-entry on same symbol within 60 minutes after close
    """

    def __init__(
        self,
        data_provider: DataManager,
        error_handler: ErrorHandler,
        reward_system: Optional[RewardSystem] = None,
        risk_manager: Optional[RiskManager] = None,
        *,
        state_size: int = 8,
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
        timeframe_entry: str = "5m",
        timeframe_setup: str = "15m",
        base_symbol: Optional[str] = None,
        notifications: Optional[object] = None,
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

        # Always simulation-mode executor per design
        self.executor = TradeExecutor(simulation_mode=True, notifications=notifications)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = ReplayBuffer(capacity=memory_size)

        self.policy_net = DQNetwork(state_size, hidden_dims, action_size).to(self.device)
        self.target_net = DQNetwork(state_size, hidden_dims, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.train_steps = 0
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(h)
        self.logger.setLevel(getattr(logging, str(getattr(config, "LOG_LEVEL", "INFO")), logging.INFO)
                            if isinstance(getattr(config, "LOG_LEVEL", "INFO"), str) else getattr(config, "LOG_LEVEL", logging.INFO))

        # Timeframes
        self.tf_entry = timeframe_entry  # "5m"
        self.tf_setup = timeframe_setup  # "15m"
        self.default_symbol = base_symbol or getattr(config, "DEFAULT_SYMBOL", "BTC/USDT")

        # Top pairs selection
        self.top_pairs = TopPairs(self.data_provider)
        self.current_universe: List[str] = [self._normalize(self.default_symbol)]
        self.last_universe_refresh: Optional[datetime] = None

        # Per-symbol “cooldown” after closing a position (no re-entry for 60 minutes)
        self.cooldown_after_close: Dict[str, datetime] = {}

        # Open position monitoring cadence (minutes)
        self.open_monitor_minutes = 1  # you can tune to 5 if you prefer
        self.last_position_check: Dict[str, datetime] = {}

        # Track last action per symbol (avoid spamming)
        self.last_action_ts: Dict[str, datetime] = {}

        # Kline backfill sizing
        self.max_request_bars = 900  # under Bybit’s 1000 limit
        self.append_increment_bars = 5  # append tiny increments, no GB downloads

    # ─────────────── Core Loop Orchestration (to be called by main.py) ─────────────── #

    def step(self, now: Optional[datetime] = None):
        """
        One “tick” of logic:
          - Periodically refresh top pairs (hourly)
          - For each symbol in universe:
            * Load incremental data (5m for entries, 15m for setup)
            * Evaluate state → select action (DQN)
            * Simulate order if action says so
            * Record experience & learn
          - Tighten stops for open positions (every minute)
        """
        now = now or datetime.now(timezone.utc)
        self._refresh_universe_if_needed(now)

        # Manage open positions frequently (minute cadence)
        self._monitor_open_positions(now)

        for sym in list(self.current_universe):
            try:
                # cooldown guard
                if sym in self.cooldown_after_close and now < self.cooldown_after_close[sym]:
                    continue

                # Build state from current market (5m + 15m features)
                state, price = self._get_state(sym)
                if state is None or price is None or price <= 0:
                    continue

                # Choose action
                action_idx = self.select_action(state)
                action_name = self._action_to_command(action_idx)

                # Throttle per-symbol actions to avoid spam (every bar)
                last_ts = self.last_action_ts.get(sym)
                if last_ts and (now - last_ts).total_seconds() < 60:  # 1 min minimal spacing
                    continue

                # Decide side and simple SL/TP skeleton
                qty = None
                sl_px, tp_px = None, None
                risk_close = False

                if action_name in ("buy", "sell"):
                    side = "long" if action_name == "buy" else "short"
                    # default 1.5 ATR stop, RR>=1.5
                    atr_val = self._safe_atr(sym)
                    if atr_val is None:
                        # fallback: 0.5% stop
                        sl_distance = 0.005 * price
                    else:
                        sl_distance = 1.5 * atr_val

                    if side == "long":
                        sl_px = max(0.0001, price - sl_distance)
                        tp_px = price + 1.5 * (price - sl_px)
                    else:
                        sl_px = price + sl_distance
                        tp_px = price - 1.5 * (sl_px - price)

                    # Risk-aware position sizing (virtual)
                    if self.risk_manager:
                        try:
                            pos = self.risk_manager.calculate_position_size(
                                symbol=sym,
                                side="long" if action_name == "buy" else "short",
                                entry_price=price,
                                stop_price=sl_px,
                                atr=atr_val,
                                rr=1.5,
                                regime=None,
                            )
                            qty = pos.position_size
                        except RiskViolationError as re:
                            self.logger.info(f"[{sym}] Risk blocked order: {re}")
                            continue

                elif action_name == "close":
                    risk_close = True

                # Execute simulated order
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

                self.last_action_ts[sym] = now

                # Register position open/close with RiskManager for bookkeeping
                status = tr.get("status")
                if status in ("open", "open_added", "open_increased") and self.risk_manager and qty:
                    # build a mirror PositionRisk snapshot (best-effort)
                    side = "long" if action_name == "buy" else "short"
                    atr_val = self._safe_atr(sym)
                    rr = 1.5
                    if atr_val is not None:
                        sl, tp = self.risk_manager.compute_sl_tp_from_atr(side, price, atr_val, rr)
                    else:
                        if side == "long":
                            sl = sl_px or (price * 0.995)
                            tp = price + rr * (price - sl)
                        else:
                            sl = sl_px or (price * 1.005)
                            tp = price - rr * (sl - price)
                    pr = self.risk_manager.calculate_position_size(
                        symbol=sym, side=side, entry_price=price, stop_price=sl, atr=atr_val, rr=rr
                    )
                    self.risk_manager.register_open_position(sym, pr)

                if status in ("closed", "closed_partial"):
                    # mark cooldown 60 minutes to avoid immediate re-entry
                    self.cooldown_after_close[sym] = now + timedelta(minutes=60)
                    # remove from risk manager
                    if self.risk_manager:
                        self.risk_manager.unregister_position(sym)

                # Build reward from immediate outcome (paper)
                entry_price = float(tr.get("entry_price") or price)
                exit_price = float(tr.get("exit_price") or entry_price)
                q_used = float(tr.get("quantity") or (qty or 0.0))

                reward_raw = self.reward_system.calculate_reward(
                    entry_price=entry_price,
                    exit_price=exit_price,
                    position_size=q_used,
                    entry_time=now,
                    exit_time=now,
                    max_drawdown=0.0,
                    volatility=0.0,
                    stop_loss_triggered=False,
                )
                reward_pts = calculate_points(
                    profit=reward_raw,
                    entry_time=now,
                    exit_time=now,
                    stop_loss_triggered=False,
                )
                reward_total = float(reward_raw + reward_pts)

                # next state
                next_state, _ = self._get_state(sym)
                done = status in ("closed",)

                if state is not None and next_state is not None:
                    self.memory.push(state, action_idx, reward_total, next_state, done)
                    self.train_steps += 1
                    self.logger.info(
                        f"[{sym}] step={self.train_steps} action={action_name} qty={q_used:.8f} "
                        f"reward={reward_total:.4f} eps={self.epsilon:.4f} status={status}"
                    )

                    if self.training:
                        self.learn()
                        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            except Exception as e:
                self.error_handler.handle(e, {"symbol": sym, "stage": "step"})

    # ─────────────── Learning ─────────────── #

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
                t_param.data.mul_((1.0 - self.tau)).add_(self.tau * p_param.data)
        except Exception as e:
            self.error_handler.log_error(e, {"stage": "learn"})

    # ─────────────── State Construction ─────────────── #

    def _get_state(self, symbol: str) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Build a compact state vector blending 15m setup & 5m entry signals.
        Uses incremental append (<= append_increment_bars) to minimize downloads.
        """
        sym = self._normalize(symbol)
        try:
            # Ensure we have recent bars (backfill small increments)
            df5 = self.data_provider.load_historical_data(sym, timeframe=self.tf_entry, max_bars=self.max_request_bars, append_increment=self.append_increment_bars)
            df15 = self.data_provider.load_historical_data(sym, timeframe=self.tf_setup, max_bars=self.max_request_bars, append_increment=self.append_increment_bars)

            if df5 is None or len(df5) == 0 or df15 is None or len(df15) == 0:
                return None, None

            # latest prices
            price = float(df5["close"].iloc[-1])

            # 5m features
            rsi5 = TechnicalIndicators.rsi(df5["close"], window=14)
            ema_fast5 = TechnicalIndicators.ema(df5["close"], window=10)
            ema_slow5 = TechnicalIndicators.ema(df5["close"], window=50)

            # 15m setup features
            rsi15 = TechnicalIndicators.rsi(df15["close"], window=14)
            ema_fast15 = TechnicalIndicators.ema(df15["close"], window=10)
            ema_slow15 = TechnicalIndicators.ema(df15["close"], window=50)

            # Normalize ratios against price where sensible
            def last(series) -> float:
                try:
                    return float(series.iloc[-1])
                except Exception:
                    return 0.0

            features = [
                price,                                  # absolute anchor (model can learn diffs)
                last(ema_fast5) / price if price else 0.0,
                last(ema_slow5) / price if price else 0.0,
                (last(rsi5) or 0.0) / 100.0,
                last(ema_fast15) / price if price else 0.0,
                last(ema_slow15) / price if price else 0.0,
                (last(rsi15) or 0.0) / 100.0,
                float(df5["volume"].iloc[-1]),
            ]
            state = np.array(features, dtype=np.float32)
            return state, price
        except Exception as e:
            self.error_handler.handle(e, {"symbol": sym, "stage": "_get_state"})
            return None, None

    def _safe_atr(self, symbol: str) -> Optional[float]:
        try:
            df = self.data_provider.load_historical_data(self._normalize(symbol), timeframe=self.tf_entry, max_bars=200, append_increment=self.append_increment_bars)
            if df is None or len(df) < 16:
                return None
            atr_series = TechnicalIndicators.atr(df["high"], df["low"], df["close"], period=14)
            if isinstance(atr_series, list):
                # list API fallback
                return float(atr_series[-1]) if atr_series and atr_series[-1] is not None else None
            return float(atr_series.iloc[-1])
        except Exception:
            return None

    # ─────────────── Universe & Monitoring ─────────────── #

    def _refresh_universe_if_needed(self, now: datetime):
        # refresh every 60 minutes
        if self.last_universe_refresh and (now - self.last_universe_refresh).total_seconds() < 3600:
            return
        try:
            pairs = self.top_pairs.get_top_pairs(
                lookback_hours_24=24,
                lookback_hours_2=2,
                limit=getattr(config, "MAX_SIMULATION_PAIRS", 5),
                include_base=self.default_symbol,
                timeframe=self.tf_setup,
            )
            # normalize like "BTC/USDT" -> "BTC/USDT"
            self.current_universe = [self._normalize(p) for p in pairs] or [self._normalize(self.default_symbol)]
            self.last_universe_refresh = now
            self.logger.info(f"[universe] refreshed: {', '.join(self.current_universe)}")
        except Exception as e:
            self.error_handler.log_error(e, {"stage": "refresh_universe"})
            if not self.current_universe:
                self.current_universe = [self._normalize(self.default_symbol)]

    def _monitor_open_positions(self, now: datetime):
        if not self.risk_manager:
            return
        for key, pos in list(self.risk_manager.open_positions.items()):
            last = self.last_position_check.get(key)
            if last and (now - last).total_seconds() < self.open_monitor_minutes * 60:
                continue
            try:
                # latest price to manage stops
                df = self.data_provider.load_historical_data(pos.symbol, timeframe=self.tf_entry, max_bars=10, append_increment=1)
                if df is None or len(df) == 0:
                    continue
                price = float(df["close"].iloc[-1])
                atr_val = self._safe_atr(pos.symbol)

                updated = self.risk_manager.dynamic_stop_management(key, price, atr=atr_val)
                self.logger.debug(f"[stop-mgr] {pos.symbol} SL→{updated.stop_loss:.6f} TP→{updated.take_profit:.6f}")
                self.last_position_check[key] = now
            except Exception as e:
                self.error_handler.log_error(e, {"symbol": pos.symbol, "stage": "monitor_open"})

    # ─────────────── Helpers ─────────────── #

    def _action_to_command(self, action_idx: int) -> str:
        # 0: buy, 1: sell, 2: hold, 3: buy (aggressive), 4: sell (aggressive), 5: close
        mapping = {0: "buy", 1: "sell", 2: "hold", 3: "buy", 4: "sell", 5: "close"}
        return mapping.get(action_idx, "hold")

    def _normalize(self, symbol: str) -> str:
        # Ensure symbols are consistent "BTC/USDT"
        s = symbol.replace("-", "/").upper()
        if "/" not in s and s.endswith("USDT"):
            base = s[:-4]
            s = f"{base}/USDT"
        return s

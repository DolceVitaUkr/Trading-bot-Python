# modules/parameter_optimization.py

import os
import pickle
import random
import logging
from typing import Dict, List, Callable, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms

import config
from modules.trade_simulator import TradeSimulator
from modules.data_manager import DataManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ParameterOptimizer:
    """
    Optimize strategy parameters via grid/random/evolutionary search.
    """

    def __init__(self):
        self.method = getattr(config, "OPTIMIZATION_METHOD", "grid_search")
        self.params = getattr(config, "OPTIMIZATION_PARAMETERS", {
            "ema_short": {"min": 5, "max": 20},
            "ema_long": {"min": 20, "max": 100},
            "rsi_period": {"min": 5, "max": 30},
            "rsi_overbought": [70, 75, 80],
            "rsi_oversold": [20, 25, 30],
        })
        # Safe default if config doesn't define it
        self.checkpoint_file = getattr(config, "OPTIMIZATION_CHECKPOINT_FILE", "optimization_checkpoint.pkl")
        self.results: pd.DataFrame = pd.DataFrame()

        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self._setup_evolutionary_tools()

    def _setup_evolutionary_tools(self):
        for key, bounds in self.params.items():
            if isinstance(bounds, dict):
                self.toolbox.register(f"attr_{key}", random.uniform, bounds["min"], bounds["max"])
            else:
                self.toolbox.register(f"attr_{key}", random.choice, bounds)

        attrs = [getattr(self.toolbox, f"attr_{k}") for k in self.params]
        self.toolbox.register("individual", tools.initCycle, creator.Individual, attrs, n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def optimize(self, objective_function: Callable[[Dict], float]) -> Dict:
        if os.path.exists(self.checkpoint_file):
            logger.info("Resuming optimization from checkpoint")
            try:
                with open(self.checkpoint_file, "rb") as f:
                    data = pickle.load(f)
                    self.results = data.get("results", self.results)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")

        if self.method == "grid_search":
            best = self._grid_search(objective_function)
        elif self.method == "random_search":
            best = self._random_search(objective_function)
        elif self.method == "evolutionary":
            best = self._evolutionary_optimization(objective_function)
        else:
            raise ValueError(f"Unknown optimization method: {self.method}")

        self._save_checkpoint()
        return best

    def run(self, symbol: Optional[str] = None, timeframe: str = "15m", initial_balance: Optional[float] = None) -> Dict:
        """
        Deletes old checkpoint and runs optimization using a simple EMA crossover objective.
        """
        if os.path.exists(self.checkpoint_file):
            try:
                os.remove(self.checkpoint_file)
                logger.info("Deleted old optimization checkpoint")
            except Exception:
                pass

        if symbol is None:
            symbol = getattr(config, "DEFAULT_SYMBOL", "BTC/USDT")

        if initial_balance is None:
            initial_balance = getattr(config, "SIMULATION_START_BALANCE", 1000.0)

        def ema_crossover_objective(params: Dict) -> float:
            try:
                dm = DataManager(test_mode=False)
                df = dm.load_historical_data(symbol, timeframe)
                if df is None or df.empty:
                    return -np.inf

                short_span = max(1, int(params.get("ema_short", 12)))
                long_span = max(short_span + 1, int(params.get("ema_long", 26)))

                df["ema_short"] = df["close"].ewm(span=short_span, adjust=False).mean()
                df["ema_long"] = df["close"].ewm(span=long_span, adjust=False).mean()

                balance, position = float(initial_balance), 0.0
                for i in range(1, len(df)):
                    prev_short, prev_long = df["ema_short"].iat[i-1], df["ema_long"].iat[i-1]
                    curr_short, curr_long = df["ema_short"].iat[i], df["ema_long"].iat[i]
                    price = df["close"].iat[i]

                    if prev_short <= prev_long and curr_short > curr_long and position == 0:
                        position = balance / price
                        balance = 0.0
                    elif prev_short >= prev_long and curr_short < curr_long and position > 0:
                        balance = position * price
                        position = 0.0

                if position > 0:
                    balance = position * df["close"].iat[-1]
                return balance - float(initial_balance)
            except Exception as e:
                logger.error(f"Objective function error with params {params}: {e}")
                return -np.inf

        best_params = self.optimize(ema_crossover_objective)
        logger.info(f"Optimization complete. Best params: {best_params}")
        return best_params

    def _grid_search(self, objective_function: Callable) -> Dict:
        combos = self._generate_parameter_combinations()
        results = []
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(self._evaluate_parameters, objective_function, c): c for c in combos}
            for f in as_completed(futures):
                try:
                    score = f.result()
                    results.append((futures[f], score))
                except Exception as e:
                    logger.warning(f"Grid search eval failed: {e}")
        return self._select_best_result(results)

    def _random_search(self, objective_function: Callable, n_iter: int = 1000) -> Dict:
        results = []
        for i in range(n_iter):
            params = {k: (random.choice(v) if isinstance(v, list) else float(np.random.uniform(v["min"], v["max"]))) for k, v in self.params.items()}
            try:
                score = self._evaluate_parameters(objective_function, params)
                results.append((params, score))
            except Exception as e:
                logger.warning(f"Random search iter {i} failed: {e}")
            if i % 100 == 0:
                self._save_checkpoint()
        return self._select_best_result(results)

    def _evolutionary_optimization(self, objective_function: Callable) -> Dict:
        self.toolbox.register("evaluate", self._ea_evaluation_wrapper, objective_function)
        pop = self.toolbox.population(n=int(getattr(config, "EA_POPULATION_SIZE", 20)))
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        algorithms.eaSimple(
            pop,
            self.toolbox,
            cxpb=float(getattr(config, "EA_CROSSOVER_PROB", 0.5)),
            mutpb=float(getattr(config, "EA_MUTATION_PROB", 0.2)),
            ngen=int(getattr(config, "EA_GENERATIONS", 10)),
            stats=stats,
            halloffame=hof,
            verbose=False
        )
        return self._decode_individual(hof[0])

    def _ea_evaluation_wrapper(self, objective_function: Callable, individual) -> tuple:
        params = self._decode_individual(individual)
        try:
            return (float(objective_function(params)),)
        except Exception as e:
            logger.warning(f"EA eval failed: {e}")
            return (-np.inf,)

    def _evaluate_parameters(self, objective_function: Callable, params: Dict) -> float:
        self._validate_parameters(params)
        return float(objective_function(params))

    def _select_best_result(self, results: List) -> Dict:
        if not results:
            raise ValueError("No valid parameter evaluations")
        df = pd.DataFrame(results, columns=["params", "score"]).sort_values("score", ascending=False).reset_index(drop=True)
        self.results = df
        return df.loc[0, "params"]

    def _generate_parameter_combinations(self) -> List[Dict]:
        total = int(np.prod([len(v) if isinstance(v, list) else 10 for v in self.params.values()]))
        if total > 10_000:
            combos = []
            for _ in range(1000):
                combo = {k: (random.choice(v) if isinstance(v, list) else float(np.random.uniform(v["min"], v["max"]))) for k, v in self.params.items()}
                combos.append(combo)
            return combos

        from itertools import product
        keys, values = zip(*[(k, v if isinstance(v, list) else list(np.linspace(v["min"], v["max"], num=10))) for k, v in self.params.items()])
        return [dict(zip(keys, prod)) for prod in product(*values)]

    def _decode_individual(self, individual) -> Dict:
        return {k: individual[i] for i, k in enumerate(self.params)}

    def _validate_parameters(self, params: Dict):
        if "ema_short" in params and "ema_long" in params:
            if params["ema_short"] >= params["ema_long"]:
                raise ValueError("ema_short must be less than ema_long")

    def _save_checkpoint(self):
        try:
            data = {"params": self.params, "results": self.results}
            with open(self.checkpoint_file, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

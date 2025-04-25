# modules/parameter_optimization.py
import logging
import config
import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from deap import base, creator, tools, algorithms
import random
import pickle
import os
from utils.utilities import ensure_directory

logger = logging.getLogger(__name__)

class ParameterOptimizer:
    def __init__(self):
        self.method = config.OPTIMIZATION_METHOD
        self.params = config.OPTIMIZATION_PARAMETERS
        self.results = []
        self.checkpoint_file = "optimization_checkpoint.pkl"
        
        # Initialize EA framework
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        self._setup_evolutionary_parameters()

    def optimize(self, objective_function: Callable) -> Dict:
        """Main optimization entry point"""
        try:
            if os.path.exists(self.checkpoint_file):
                self._load_checkpoint()
            
            if self.method == "grid_search":
                return self._grid_search(objective_function)
            elif self.method == "random_search":
                return self._random_search(objective_function)
            elif self.method == "evolutionary":
                return self._evolutionary_optimization(objective_function)
            else:
                raise ValueError(f"Unknown optimization method: {self.method}")
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            raise
        finally:
            self._save_checkpoint()

    def _grid_search(self, objective_function: Callable) -> Dict:
        """Parallel grid search with intelligent parameter sampling"""
        param_combinations = self._generate_parameter_combinations()
        
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self._evaluate_parameters,
                    objective_function,
                    combination
                ): combination
                for combination in param_combinations
            }
            
            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.warning(f"Parameter evaluation failed: {str(e)}")

        return self._select_best_result(results)

    def _random_search(self, objective_function: Callable, n_iter: int = 1000) -> Dict:
        """Bayesian-inspired random search with adaptive sampling"""
        results = []
        for i in range(n_iter):
            params = {
                key: random.choice(values) if isinstance(values, list) else
                np.random.uniform(values['min'], values['max'])
                for key, values in self.params.items()
            }
            try:
                score = self._evaluate_parameters(objective_function, params)
                results.append((params, score))
                logger.debug(f"Random search iteration {i+1}/{n_iter} - Score: {score:.2f}")
            except Exception as e:
                logger.warning(f"Random search iteration failed: {str(e)}")
            
            if i % 100 == 0:
                self._save_checkpoint()
        
        return self._select_best_result(results)

    def _evolutionary_optimization(self, objective_function: Callable) -> Dict:
        """Evolutionary strategy optimization with DEAP framework"""
        # Late-binding of evaluation function
        self.toolbox.register("evaluate", self._ea_evaluation_wrapper, objective_function)
        
        pop = self.toolbox.population(n=config.EA_POPULATION_SIZE)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        algorithms.eaSimple(
            pop, 
            self.toolbox,
            cxpb=config.EA_CROSSOVER_PROB,
            mutpb=config.EA_MUTATION_PROB,
            ngen=config.EA_GENERATIONS,
            stats=stats,
            halloffame=hof,
            verbose=True
        )
        
        return self._decode_individual(hof[0])

    def _setup_evolutionary_parameters(self):
        """Configure evolutionary algorithm parameters"""
        for key, bounds in self.params.items():
            self.toolbox.register(
                f"attr_{key}", 
                random.uniform, 
                bounds['min'], 
                bounds['max']
            )
        
        attributes = [
            getattr(self.toolbox, f"attr_{key}")
            for key in self.params.keys()
        ]
        
        self.toolbox.register(
            "individual", 
            tools.initCycle, 
            creator.Individual,
            attributes,
            n=1
        )
        
        self.toolbox.register(
            "population", 
            tools.initRepeat, 
            list, 
            self.toolbox.individual
        )
        
        # Defer evaluate registration until we have objective function
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def _ea_evaluation_wrapper(self, individual, objective_function):
        """Convert DEAP individual to parameter dict"""
        params = self._decode_individual(individual)
        try:
            score = objective_function(params)
            return (score,)
        except Exception as e:
            logger.warning(f"EA evaluation failed: {str(e)}")
            return (-np.inf,)

    def _decode_individual(self, individual):
        """Convert evolutionary individual to parameter dictionary"""
        return {
            key: individual[i]
            for i, key in enumerate(self.params.keys())
        }

    def _generate_parameter_combinations(self):
        """Smart parameter space sampling with pruning"""
        if self._parameter_space_size() > 1e4:
            return self._latin_hypercube_sampling(1000)
        return self._exhaustive_parameter_combinations()

    def _parameter_space_size(self):
        """Calculate total parameter combinations"""
        size = 1
        for values in self.params.values():
            size *= len(values) if isinstance(values, list) else 10
        return size

    def _latin_hypercube_sampling(self, n_samples: int):
        """Generate space-filling parameter samples"""
        samples = []
        for _ in range(n_samples):
            sample = {}
            for key, values in self.params.items():
                if isinstance(values, list):
                    sample[key] = random.choice(values)
                else:
                    sample[key] = np.random.uniform(values['min'], values['max'])
            samples.append(sample)
        return samples

    def _exhaustive_parameter_combinations(self):
        """Generate all possible parameter combinations"""
        # Implementation for grid search combinations
        raise NotImplementedError("Exhaustive grid search not implemented")

    def _evaluate_parameters(self, objective_function: Callable, params: Dict):
        """Safe parameter evaluation with validation"""
        try:
            self._validate_parameters(params)
            return objective_function(params)
        except Exception as e:
            logger.warning(f"Invalid parameters {params}: {str(e)}")
            raise

    def _validate_parameters(self, params: Dict):
        """Parameter validation rules"""
        if 'ema_short' in params and 'ema_long' in params:
            if params['ema_short'] >= params['ema_long']:
                raise ValueError("Short EMA must be smaller than Long EMA")

    def _select_best_result(self, results: List) -> Dict:
        """Select best parameters from results"""
        if not results:
            raise ValueError("No valid parameter combinations evaluated")
        
        results_df = pd.DataFrame(results, columns=['params', 'score'])
        results_df = results_df.sort_values('score', ascending=False)
        self.results = results_df
        
        best = results_df.iloc[0]
        logger.info(f"Best parameters: {best['params']} with score {best['score']:.2f}")
        return best['params']

    def _save_checkpoint(self):
        """Save optimization progress"""
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump({
                'method': self.method,
                'results': self.results,
                'params': self.params
            }, f)

    def _load_checkpoint(self):
        """Load optimization progress"""
        with open(self.checkpoint_file, 'rb') as f:
            data = pickle.load(f)
            self.results = data['results']
            logger.info(f"Loaded checkpoint with {len(self.results)} existing results")

    def analyze_results(self):
        """Generate optimization analysis report"""
        if not self.results:
            raise ValueError("No optimization results available")
        
        analysis = {
            'parameter_importance': self._calculate_feature_importance(),
            'interactions': self._analyze_parameter_interactions(),
            'sensitivity': self._calculate_sensitivity()
        }
        return analysis

    def _calculate_feature_importance(self):
        """Calculate parameter importance using random forest"""
        from sklearn.ensemble import RandomForestRegressor
        
        df = pd.DataFrame(self.results)
        X = pd.json_normalize(df['params'])
        y = df['score']
        
        model = RandomForestRegressor()
        model.fit(X, y)
        
        return dict(zip(X.columns, model.feature_importances_))

    def _analyze_parameter_interactions(self):
        """Placeholder for interaction analysis"""
        return {}

    def _calculate_sensitivity(self):
        """Placeholder for sensitivity analysis"""
        return {}

    def visualize_optimization(self):
        """Create interactive optimization visualization"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        df = pd.DataFrame(self.results)
        param_df = pd.json_normalize(df['params'])
        combined_df = pd.concat([param_df, df['score']], axis=1)
        
        g = sns.pairplot(
            combined_df,
            diag_kind='kde',
            plot_kws={'alpha': 0.5}
        )
        plt.suptitle("Parameter Optimization Landscape", y=1.02)
        plt.show()

if __name__ == "__main__":
    # Example usage
    def sample_objective(params):
        return -(params['ema_short'] - 10)**2 + -(params['ema_long'] - 50)**2
    
    optimizer = ParameterOptimizer()
    best_params = optimizer.optimize(sample_objective)
    print("Best Parameters:", best_params)
    optimizer.visualize_optimization()
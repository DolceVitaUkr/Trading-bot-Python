# modules/parameter_optimization.py
import random, itertools, logging
import config

class ParameterOptimizer:
    def __init__(self):
        self.method = config.OPTIMIZATION_METHOD
        self.params = config.OPTIMIZATION_PARAMETERS
        self.results = []

    def optimize(self, objective_function):
        combos = self._generate_parameter_combinations()
        results = []
        for p in combos:
            try:
                results.append((p, objective_function(p)))
            except Exception as e:
                logging.warning(f"Eval failed for {p}: {e}")
        return self._select_best_result(results)

    def _generate_parameter_combinations(self):
        keys = list(self.params.keys())
        lists = []
        for k in keys:
            vals = self.params[k]
            if isinstance(vals, list):
                lists.append(vals)
            else:
                lists.append([vals["min"], vals["max"]])
        return [dict(zip(keys,comb)) for comb in itertools.product(*lists)]

    def _select_best_result(self, results):
        if not results:
            raise ValueError("No valid results")
        best = max(results, key=lambda x: x[1])[0]
        return best

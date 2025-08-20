"""Parameter optimisation placeholder."""

from __future__ import annotations

from typing import Callable, Dict, Any


class Optimizer:
    """A very small hook that simply evaluates a fitness function."""

    def optimise(self, func: Callable[[Dict[str, Any]], float], param: Dict[str, Any]) -> float:
        return func(param)


__all__ = ["Optimizer"]

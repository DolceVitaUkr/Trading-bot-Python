from __future__ import annotations

from typing import Dict, List


class DiffManager:
    """Track proposed and confirmed trading actions for assets."""

    def __init__(self) -> None:
        self._proposals: Dict[str, List[str]] = {}
        self._applied: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    def set_proposed_actions(self, asset: str, actions: List[str]) -> None:
        """Register ``actions`` as the current proposal for ``asset``."""

        self._proposals[asset] = list(actions)

    # ------------------------------------------------------------------
    def proposed_actions(self, asset: str) -> List[str]:
        """Return pending proposed actions for ``asset``."""

        return self._proposals.get(asset, [])

    # ------------------------------------------------------------------
    def confirm_actions(self, asset: str) -> List[str]:
        """Mark the proposal for ``asset`` as applied and return it."""

        actions = self._proposals.pop(asset, [])
        self._applied[asset] = actions
        return actions

    # ------------------------------------------------------------------
    def applied_actions(self, asset: str) -> List[str]:
        """Return the actions previously confirmed for ``asset``."""

        return self._applied.get(asset, [])


__all__ = ["DiffManager"]

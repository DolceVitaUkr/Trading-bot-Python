# file: core/runtimecontroller.py
"""RuntimeController compatibility layer with underscore-free API."""

from __future__ import annotations

from .runtime_controller import RuntimeController as _RuntimeController


class RuntimeController(_RuntimeController):
    """Expose new naming conventions for runtime control operations."""

    def start(self) -> None:
        """Mark the runtime as started."""
        self.state.setdefault("global", {})["running"] = True
        self._save()

    def stop(self) -> None:
        """Mark the runtime as stopped."""
        self.state.setdefault("global", {})["running"] = False
        self._save()

    def enablelive(self, asset: str) -> bool:
        """Enable live trading for ``asset`` after validation."""
        super().enable_live(asset)
        return True

    def disablelive(self, asset: str, closeonly: bool = False) -> None:
        """Disable live trading for ``asset``."""
        super().disable_live(asset, close_only=closeonly)

    def setglobalkill(self, active: bool) -> None:
        """Toggle the global kill switch."""
        super().set_global_kill(active)

    def recordtraderesult(self, asset: str, islive: bool, pnlafterfees: float) -> None:
        """Record the result of a trade for loss tracking."""
        super().record_trade_result(asset, is_live=islive, pnl_after_fees=pnlafterfees)

    def hourlypaperrecap(self) -> None:
        """Placeholder for hourly paper recap notifications."""
        self.notifier.send("Hourly paper recap not implemented")

    def getstate(self) -> dict:
        """Return the current runtime state."""
        return super().get_state()


__all__ = ["RuntimeController"]

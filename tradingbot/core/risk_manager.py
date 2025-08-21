# file: core/risk_manager.py
"""Compatibility wrapper for riskmanager module."""

from .riskmanager import RiskManager, OrderProposal, Position

__all__ = ["RiskManager", "OrderProposal", "Position"]

# modules/rollout_manager.py
from __future__ import annotations
from typing import Dict, Optional
import logging

import config
from modules.runtime_state import RuntimeState

logger = logging.getLogger(__name__)


class RolloutManager:
    """
    Rollout stage gatekeeper + helper utilities.

    Stages (from readme):
      1: Crypto Paper (Learn)
      2: Crypto Live (+ Exploration)
      3: Crypto Live + Perps/Forex Paper
      4: Crypto & Forex Live (+ Options Paper)
      5: Full Rollout (Crypto Live + Forex Live; Options optional)

    Domain mapping:
      - crypto: paper allowed from stage 1; live from stage 2
      - perps:  paper allowed from stage 3; live (when enabled later) can share rules with crypto (spot+perp)
      - forex:  paper from stage 3; live from stage 4
      - options: paper from stage 4; live from stage 5 (optional)

    This class does NOT place orders. It only decides if features are permitted,
    reconciles toggles on boot, and updates persistent state.
    """

    def __init__(self, state: RuntimeState):
        self.state = state

        # Gate definitions
        self._paper_gate = {
            "crypto": 1,
            "perps": 3,
            "forex": 3,
            "options": 4,
        }
        self._live_gate = {
            "crypto": 2,
            # perps live rides under crypto EXCHANGE_PROFILE = "spot+perp"
            "forex": 4,
            "options": 5,  # optional even at 5; still gated
        }

    # ──────────────────────────────────────────────────────────────────────
    # Capability checks
    # ──────────────────────────────────────────────────────────────────────
    def allows_paper(self, domain: str) -> bool:
        stage = self.state.get_stage()
        return stage >= self._paper_gate.get(domain, 99)

    def allows_live(self, domain: str) -> bool:
        stage = self.state.get_stage()
        need = self._live_gate.get(domain, 99)
        return stage >= need

    def can_trade_crypto_live(self) -> bool:
        return self.allows_live("crypto")

    def can_trade_forex_live(self) -> bool:
        return self.allows_live("forex")

    def can_trade_options_live(self) -> bool:
        return self.allows_live("options")

    # ──────────────────────────────────────────────────────────────────────
    # Reconciliation on boot
    # ──────────────────────────────────────────────────────────────────────
    def reconcile_on_boot(self) -> None:
        """
        Enforce safe-restart semantics:
          - Forex and Options toggles OFF on boot (no new orders) but monitor/reconcile open positions.
          - Crypto retains last state; Perps governed by EXCHANGE_PROFILE (spot|perp|spot+perp).
          - Paper wallets persist; never reset unless user requests.
        """
        # FX/Options: disable trading toggles at boot, but keep monitoring
        self.state.set_flag("forex_enabled", 0)   # OFF for new orders
        self.state.set_flag("options_enabled", 0) # OFF for new orders

        # Ensure we have stage & baseline exploration min
        if self.state.get_flag("exploration_min_rate") is None:
            self.state.set_flag("exploration_min_rate", config.EXPLORATION_MIN_RATE)

        # Record current profile for crypto (spot/perp/spot+perp)
        self.state.set_flag("exchange_profile", config.EXCHANGE_PROFILE)

        self.state._event("rollout.reconcile_boot", "Toggles normalized at boot")
        self.state.save()

    # ──────────────────────────────────────────────────────────────────────
    # Guard helpers (returns bool and optionally pushes a UI message)
    # ──────────────────────────────────────────────────────────────────────
    def guard_paper(self, domain: str, ui=None) -> bool:
        if not self.allows_paper(domain):
            msg = f"Stage gate: Paper trading for {domain} not allowed at stage {self.state.get_stage()}."
            logger.warning(msg)
            if ui:
                ui.push_warning(msg)
            return False
        return True

    def guard_live(self, domain: str, ui=None) -> bool:
        if not self.allows_live(domain):
            msg = f"Stage gate: Live trading for {domain} not allowed at stage {self.state.get_stage()}."
            logger.warning(msg)
            if ui:
                ui.push_warning(msg)
            return False
        return True

    # ──────────────────────────────────────────────────────────────────────
    # Stage transitions (manual promotion via UI, if you expose it)
    # ──────────────────────────────────────────────────────────────────────
    def set_stage(self, stage: int) -> None:
        """Force set stage (e.g., after KPI review) — call from a UI handler with confirmation."""
        prev = self.state.get_stage()
        self.state.set_stage(stage)
        self.state._event("rollout.stage_set", f"{prev} -> {stage}")
        self.state.save()

    # ──────────────────────────────────────────────────────────────────────
    # Domain enable helpers
    # ──────────────────────────────────────────────────────────────────────
    def enable_crypto_live(self) -> bool:
        if not self.guard_live("crypto"):
            return False
        self.state.set_domain_live("crypto", True)
        self.state._event("domain.crypto.live_on", "")
        self.state.save()
        return True

    def disable_crypto_live(self) -> None:
        self.state.set_domain_live("crypto", False)
        self.state._event("domain.crypto.live_off", "")
        self.state.save()

    def enable_forex_live(self) -> bool:
        if not self.guard_live("forex"):
            return False
        self.state.set_flag("forex_enabled", 1)
        self.state._event("domain.forex.live_on", "")
        self.state.save()
        return True

    def disable_forex_live(self) -> None:
        self.state.set_flag("forex_enabled", 0)
        self.state._event("domain.forex.live_off", "")
        self.state.save()

    def enable_options_live(self) -> bool:
        if not self.guard_live("options"):
            return False
        self.state.set_flag("options_enabled", 1)
        self.state._event("domain.options.live_on", "")
        self.state.save()
        return True

    def disable_options_live(self) -> None:
        self.state.set_flag("options_enabled", 0)
        self.state._event("domain.options.live_off", "")
        self.state.save()

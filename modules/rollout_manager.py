# modules/rollout_manager.py
from typing import Optional, Dict, Any
from modules.runtime_state import RuntimeState

class BalanceAdapter:
    """
    Interface each venue/domain must implement.
    Implementations:
      - BybitSpotBalanceAdapter
      - BybitPerpBalanceAdapter
      - ForexBrokerBalanceAdapter
      - OptionsBrokerBalanceAdapter
    """
    def available_balance(self) -> float:
        raise NotImplementedError  # MUST be implemented against real API

class RolloutManager:
    """
    Gatekeeper for rollout stages, domain toggles, canary ramp, wallet segregation,
    exposure caps, and KPI gates. No mocks; live balances must come from adapters.
    """
    def __init__(self, state: RuntimeState, adapters: Dict[str, BalanceAdapter]):
        """
        adapters: dict keyed by domain {"crypto": ..., "perp": ..., "forex": ..., "options": ...}
        Each value must implement available_balance() hitting the real venue/broker API.
        """
        self.state = state
        self.adapters = adapters
        self.state.load()

    # ── Startup ────────────────────────────────────────────────────────────────
    def startup_sequence(self) -> None:
        stage = self.state.get("rollout_stage")
        print(f"[Rollout] Booting at Stage {stage}")

        if stage == 1:
            self._enable_domains(crypto_spot=True, perps=False, forex=False, options=False, live=False)
        elif stage == 2:
            self._enable_domains(crypto_spot=True, perps=False, forex=False, options=False, live=True)
        elif stage == 3:
            self._enable_domains(crypto_spot=True, perps=True, forex=False, options=False, live=True)
        elif stage == 4:
            self._enable_domains(crypto_spot=True, perps=True, forex=True, options=False, live=True)
        elif stage == 5:
            self._enable_domains(crypto_spot=True, perps=True, forex=True, options=True, live=True)

        self._ensure_stage_wallets(stage)

    # ── Stage control ──────────────────────────────────────────────────────────
    def set_stage(self, new_stage: int) -> None:
        if not (1 <= new_stage <= 5):
            raise ValueError("Rollout stage must be between 1 and 5.")
        self.state.set_rollout_stage(new_stage)
        self.startup_sequence()
        print(f"[Rollout] Stage updated to {new_stage}")

    def _ensure_stage_wallets(self, stage: int) -> None:
        if stage >= 1:
            self.state.ensure_paper_wallet("Crypto_Paper", initial=1000.0)
        if stage >= 3:
            self.state.ensure_paper_wallet("Perps_Paper", initial=1000.0)
        if stage >= 3:
            self.state.ensure_paper_wallet("Forex_Paper", initial=1000.0)
        if stage >= 4:
            self.state.ensure_paper_wallet("ForexOptions_Paper", initial=1000.0)

    def _enable_domains(self, crypto_spot: bool, perps: bool, forex: bool, options: bool, live: bool) -> None:
        self.state.set_domain("crypto", "enabled", crypto_spot)
        self.state.set_domain("crypto", "live", live and crypto_spot)

        self.state.set_domain("perp", "enabled", perps)
        self.state.set_domain("perp", "live", live and perps)

        self.state.set_domain("forex", "enabled", forex)
        self.state.set_domain("forex", "live", live and forex)

        self.state.set_domain("options", "enabled", options)
        self.state.set_domain("options", "live", live and options)

    # ── Canary (trade-count/time gated ramp) ───────────────────────────────────
    def start_canary_for_domain(self, domain: str) -> None:
        self.state.start_canary(domain)

    def get_current_canary_pct(self) -> Optional[float]:
        return self.state.canary_step_pct()

    def register_canary_trade(self) -> None:
        self.state.increment_canary_trade()
        self.state.advance_canary_if_needed()

    # ── Guardrails ─────────────────────────────────────────────────────────────
    def check_kpi_gate(self, domain: str) -> bool:
        kpi = self.state.get("kpi").get(domain, {})
        if not kpi:
            return False
        return (
            kpi.get("win_rate", 0) >= 0.65 and
            kpi.get("sharpe", 0)   >= 1.5 and
            kpi.get("max_dd", 1)   <= 0.20
        )

    def exposure_cap_for_domain(self, domain: str) -> float:
        stage = self.state.get("rollout_stage")
        if domain == "crypto":
            return 0.10 if stage >= 2 else 0.0
        if domain == "perp":
            return 0.15 if stage >= 3 else 0.0
        if domain == "forex":
            return 0.10 if stage >= 4 else 0.0
        if domain == "options":
            return 0.05 if stage >= 5 else 0.0
        return 0.0

    # ── Balances / segregation ─────────────────────────────────────────────────
    def available_balance(self, domain: str, live: bool) -> float:
        """
        Live: MUST use adapter.available_balance() (real API call).
        Paper: returns segregated paper wallet balance from runtime state.
        """
        if live:
            adapter = self.adapters.get(domain)
            if adapter is None:
                raise RuntimeError(f"No balance adapter configured for domain '{domain}'.")
            return float(adapter.available_balance())

        paper_name = self._paper_wallet_name(domain)
        return float(self.state.get("paper_wallets", {}).get(paper_name, {}).get("balance", 0.0))

    def record_paper_balance(self, domain: str, balance: float) -> None:
        self.state.set_paper_balance(self._paper_wallet_name(domain), float(balance))

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _paper_wallet_name(self, domain: str) -> str:
        mapping = {
            "crypto": "Crypto_Paper",
            "perp": "Perps_Paper",
            "forex": "Forex_Paper",
            "options": "ForexOptions_Paper",
        }
        try:
            return mapping[domain]
        except KeyError:
            raise ValueError(f"Unknown domain '{domain}'")

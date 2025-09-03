from typing import Optional
import os

BASE_CCY = os.environ.get("BASE_CCY", "USD").upper()

def get_rate(pair: str) -> Optional[float]:
    """Stub for FX rate lookup. Replace with IBKR quotes.
    Returns None if unknown.
    """
    return None

def convert_value(amount: float, ccy: str, base: str = BASE_CCY) -> float:
    ccy = (ccy or base).upper()
    base = (base or BASE_CCY).upper()
    if ccy == base:
        return float(amount)
    pair = f"{ccy}.{base}"  # IBKR-style, e.g., EUR.USD
    rate = get_rate(pair)
    if rate is None or rate <= 0:
        return float(amount)  # fallback: no conversion
    return float(amount) * rate
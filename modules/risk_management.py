# Default risk parameters
DEFAULT_STOP_LOSS_pct = 0.15   # 15% stop-loss by default
MAX_STOP_LOSS_pct = 0.20       # allow up to 20% stop-loss if loosened

def calculate_stop_loss_price(entry_price: float, side: str, stop_loss_pct: float = DEFAULT_STOP_LOSS_pct) -> float:
    """
    Calculate the stop-loss price for a given entry price and side of the position.
    stop_loss_pct is clamped between DEFAULT_STOP_LOSS_pct and MAX_STOP_LOSS_pct.
    For a long position, this returns entry_price * (1 - stop_loss_pct).
    For a short position, this returns entry_price * (1 + stop_loss_pct).
    """
    # Enforce the maximum cap on stop_loss_pct
    if stop_loss_pct > MAX_STOP_LOSS_pct:
        stop_loss_pct = MAX_STOP_LOSS_pct
    if stop_loss_pct < 0:
        stop_loss_pct = 0  # no negative percentages
    side = side.lower()
    if side in ('long', 'buy'):
        # Long position stop price (below entry)
        return entry_price * (1 - stop_loss_pct)
    elif side in ('short', 'sell'):
        # Short position stop price (above entry)
        return entry_price * (1 + stop_loss_pct)
    else:
        raise ValueError(f"Unknown position side: {side}")

def check_stop_loss(position: dict, current_price: float, stop_loss_pct: float = DEFAULT_STOP_LOSS_pct) -> bool:
    """
    Check if the position's unrealized loss has reached the stop-loss threshold.
    Returns True if the stop-loss should trigger (i.e., the position should be closed to prevent further loss).
    - position: a dict with at least 'entry_price' and 'side' keys (as used in Exchange.positions).
    - current_price: the latest market price of the asset.
    - stop_loss_pct: the percentage loss threshold (will be capped to MAX_STOP_LOSS_pct).
    """
    if position is None:
        return False  # No position to check
    entry_price = position.get('entry_price')
    side = position.get('side')
    if entry_price is None or side is None:
        return False
    # Calculate stop-loss price level for this position
    stop_price = calculate_stop_loss_price(entry_price, side, stop_loss_pct)
    side = side.lower()
    if side in ('long', 'buy'):
        # For long: trigger if current price falls to or below the stop price
        return current_price <= stop_price
    elif side in ('short', 'sell'):
        # For short: trigger if current price rises to or above the stop price
        return current_price >= stop_price
    else:
        return False

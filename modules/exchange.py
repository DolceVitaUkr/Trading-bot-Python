import requests

class Exchange:
    def __init__(self, use_testnet: bool = False):
        """
        Initialize the Exchange with Bybit mainnet for market data.
        use_testnet is ignored to enforce real market data usage.
        """
        # Always use real Bybit endpoint for data (ignore testnet)&#8203;:contentReference[oaicite:1]{index=1}
        self.base_url = "https://api.bybit.com"
        self.positions = {}  # track open positions by symbol (virtual trading)

    def get_price(self, symbol: str) -> float:
        """
        Fetch the latest market price for the given symbol from Bybit.
        """
        endpoint = "/v5/market/tickers"
        # Use linear (USDT perpetual) category by default for futures; adjust if needed per symbol.
        params = {"category": "linear", "symbol": symbol}
        response = requests.get(self.base_url + endpoint, params=params)
        data = response.json()
        try:
            last_price = data["result"]["list"][0]["lastPrice"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Failed to get price for {symbol}: {data}")
        return float(last_price)

    def create_order(self, symbol: str, side: str, quantity: float = 1.0, price: float = None):
        """
        Simulate creating an order (market order by default) on the exchange.
        This will open or modify a virtual position instead of executing a real trade.
        """
        side = side.lower()
        # Determine execution price: use provided price or fetch current market price
        exec_price = price if price is not None else self.get_price(symbol)
        position = self.positions.get(symbol)

        if position:
            # If a position already exists on this symbol, adjust or close it based on the new order
            current_side = position['side']
            # Determine if this order is same direction or opposite
            if (current_side == 'long' and side in ('sell', 'short')) or (current_side == 'short' and side in ('buy', 'long')):
                # Opposite side order -> treat as closing or reducing position
                if quantity >= position['quantity']:
                    # Close the existing position fully (and reverse if extra quantity remains)
                    closed_qty = position['quantity']
                    self.positions[symbol] = None  # position closed
                    # If order quantity is larger, open a new position in opposite direction with remaining qty
                    remaining_qty = quantity - closed_qty
                    if remaining_qty > 0:
                        new_side = 'long' if side in ('buy', 'long') else 'short'
                        self.positions[symbol] = {
                            'symbol': symbol,
                            'side': new_side,
                            'quantity': remaining_qty,
                            'entry_price': exec_price
                        }
                    else:
                        # Fully closed, no new position
                        del self.positions[symbol]
                else:
                    # Partial close: reduce the existing position's quantity
                    position['quantity'] -= quantity
                    self.positions[symbol] = position
                return self.positions.get(symbol)
            else:
                # Same direction order -> increase position (average the entry price)
                new_total_qty = position['quantity'] + quantity
                # Calculate volume-weighted average price for the new total position
                avg_price = ((position['entry_price'] * position['quantity']) + (exec_price * quantity)) / new_total_qty
                position['entry_price'] = avg_price
                position['quantity'] = new_total_qty
                self.positions[symbol] = position
                return position
        else:
            # No open position for this symbol, so open a new one
            new_side = 'long' if side in ('buy', 'long') else 'short'
            self.positions[symbol] = {
                'symbol': symbol,
                'side': new_side,
                'quantity': quantity,
                'entry_price': exec_price
            }
            return self.positions[symbol]

    def get_position(self, symbol: str):
        """
        Get the current open position for a symbol, or None if no position is open.
        """
        return self.positions.get(symbol)

    def close_position(self, symbol: str):
        """
        Simulate closing the position on the given symbol at current market price.
        Returns the details of the closed position (including P&L).
        """
        position = self.positions.get(symbol)
        if not position:
            return None  # no position to close
        # Use current market price as exit
        exit_price = self.get_price(symbol)
        entry_price = position['entry_price']
        side = position['side']
        qty = position['quantity']
        # Calculate profit/loss
        if side == 'long':
            pnl = (exit_price - entry_price) * qty
        else:  # short
            pnl = (entry_price - exit_price) * qty
        # Prepare closed position report
        closed_position = {
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': qty,
            'pnl': pnl
        }
        # Remove the position from open positions
        del self.positions[symbol]
        return closed_position

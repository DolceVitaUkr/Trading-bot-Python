# test.py
import ccxt
import config

exchange = ccxt.bybit({
    'apiKey': config.BYBIT_API_KEY,
    'secret': config.BYBIT_API_SECRET,
    'options': {
        'defaultType': 'contract',
        'accountsByType': {
            'unified': 'UNIFIED'
        }
    }
})

# Verify connectivity
print(exchange.fetch_balance(params={'accountType': 'UNIFIED'}))
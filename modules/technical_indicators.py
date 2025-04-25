def moving_average(data, period: int):
    """
    Calculate the Simple Moving Average (SMA) of the data over the given period.
    Returns the average of the last 'period' values.
    """
    if data is None or period <= 0 or len(data) < period:
        return None
    return sum(data[-period:]) / period

def exponential_moving_average(data, period: int):
    """
    Calculate the Exponential Moving Average (EMA) of the data over the given period.
    Returns the EMA of the data series.
    """
    if data is None or period <= 0 or len(data) < period:
        return None
    # Start with a simple moving average for the first 'period' points
    ema = sum(data[:period]) / period
    multiplier = 2 / (period + 1)
    for price in data[period:]:
        ema = (price - ema) * multiplier + ema
    return ema

def adx(high, low, close, period: int = 14):
    """
    Calculate the Average Directional Index (ADX) over the given period.
    ADX reflects trend strength by combining +DI and -DI.
    Returns the latest ADX value.
    """
    n = period
    if high is None or low is None or close is None or len(high) < n+1 or len(low) < n+1 or len(close) < n+1:
        return None
    length = len(close)
    # Initialize arrays for True Range (TR), +DM, -DM
    tr = [0.0] * length
    pdm = [0.0] * length  # positive directional movement
    ndm = [0.0] * length  # negative directional movement
    for i in range(1, length):
        # Calculate True Range
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        # Calculate directional movement
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        pdm[i] = up_move if up_move > down_move and up_move > 0 else 0.0
        ndm[i] = down_move if down_move > up_move and down_move > 0 else 0.0
    # Smooth the sums of TR, +DM, -DM over 'n' periods using Wilder's smoothing
    tr_sum = sum(tr[1:n+1])
    pdm_sum = sum(pdm[1:n+1])
    ndm_sum = sum(ndm[1:n+1])
    # Calculate initial Directional Indices (DI)
    if tr_sum == 0:
        plus_di = 0.0
        minus_di = 0.0
    else:
        plus_di = (pdm_sum / tr_sum) * 100
        minus_di = (ndm_sum / tr_sum) * 100
    # Calculate initial DX value at period index
    if plus_di + minus_di == 0:
        dx = 0.0
    else:
        dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    # Initialize ADX list (only necessary from 2n-1 onward)
    adx_list = [None] * length
    dx_list = [None] * length
    dx_list[n] = dx
    # Compute DX for each subsequent time step
    for i in range(n+1, length):
        # Update smoothed TR and DM sums (Wilder's smoothing)
        tr_sum = tr_sum - (tr_sum / n) + tr[i]
        pdm_sum = pdm_sum - (pdm_sum / n) + pdm[i]
        ndm_sum = ndm_sum - (ndm_sum / n) + ndm[i]
        # Recalculate DI and DX for this index
        if tr_sum == 0:
            plus_di = 0.0
            minus_di = 0.0
        else:
            plus_di = (pdm_sum / tr_sum) * 100
            minus_di = (ndm_sum / tr_sum) * 100
        if plus_di + minus_di == 0:
            dx_list[i] = 0.0
        else:
            dx_list[i] = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    # Compute initial ADX as the average of the first 'n' DX values (from index n to 2n-1)
    if length < 2 * n:
        # Not enough data to compute full ADX
        return dx_list[n]
    initial_adx = sum([dx_val for dx_val in dx_list[n:2*n] if dx_val is not None]) / n
    adx_list[2*n - 1] = initial_adx
    # Compute ADX for subsequent points using Wilder's smoothing on DX
    for j in range(2*n, length):
        prev_adx = adx_list[j-1] if adx_list[j-1] is not None else initial_adx
        adx_list[j] = ((prev_adx * (n - 1)) + (dx_list[j] if dx_list[j] is not None else 0.0)) / n
    # Return the latest ADX value
    return adx_list[-1]

def cci(high, low, close, period: int = 20):
    """
    Calculate the Commodity Channel Index (CCI) for the given period.
    CCI measures how far the price is from its statistical mean.
    Returns the latest CCI value.
    """
    if high is None or low is None or close is None or len(close) < period:
        return None
    # Typical Price (TP) for each period = (High + Low + Close) / 3
    tp_series = [(high[i] + low[i] + close[i]) / 3.0 for i in range(len(close))]
    # Calculate SMA of TP over the last 'period' values
    tp_window = tp_series[-period:]
    sma_tp = sum(tp_window) / period
    # Calculate mean deviation of TP over the period
    mean_dev = sum(abs(tp - sma_tp) for tp in tp_window) / period
    if mean_dev == 0:
        return 0.0  # avoid division by zero; CCI is zero if prices haven't moved
    # CCI formula: (TP_latest - SMA_TP) / (0.015 * mean_dev)
    cci_value = (tp_series[-1] - sma_tp) / (0.015 * mean_dev)
    return cci_value

def williams_r(high, low, close, period: int = 14):
    """
    Calculate the Williams %R indicator over the given period.
    Williams %R shows the level of the close relative to the highest high and lowest low of the period.
    Returns the latest %R value (between 0 and -100).
    """
    if high is None or low is None or close is None or len(close) < period:
        return None
    highest_high = max(high[-period:])
    lowest_low = min(low[-period:])
    if highest_high == lowest_low:
        return 0.0  # avoid division by zero if all prices equal in the period
    # %R formula: (HighestHigh - LastClose) / (HighestHigh - LowestLow) * -100
    percent_r = (highest_high - close[-1]) / (highest_high - lowest_low) * -100
    return percent_r

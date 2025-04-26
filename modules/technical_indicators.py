# modules/technical_indicators.py
from typing import List, Optional
import numpy as np
import pandas as pd

# Stand-alone functions your tests expect:

def moving_average(data: List[float], period: int) -> Optional[float]:
    if data is None or period <= 0 or len(data) < period:
        return None
    return sum(data[-period:]) / period

def exponential_moving_average(data: List[float], period: int) -> Optional[float]:
    if data is None or period <= 0 or len(data) < period:
        return None
    ema = sum(data[:period]) / period
    mult = 2 / (period + 1)
    for price in data[period:]:
        ema = (price - ema) * mult + ema
    return ema

def adx(high: List[float], low: List[float], close: List[float], period: int = 14) -> Optional[float]:
    n = period
    if not (high and low and close) or len(close) < 2*n:
        return None
    length = len(close)
    tr = [0.0]*length
    pdm = [0.0]*length
    ndm = [0.0]*length
    for i in range(1, length):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
        up = high[i]-high[i-1]; down = low[i-1]-low[i]
        pdm[i] = up if up>down and up>0 else 0.0
        ndm[i] = down if down>up and down>0 else 0.0
    tr_sum = sum(tr[1:n+1]); pdm_sum = sum(pdm[1:n+1]); ndm_sum = sum(ndm[1:n+1])
    plus = (pdm_sum/tr_sum*100) if tr_sum else 0.0
    minus = (ndm_sum/tr_sum*100) if tr_sum else 0.0
    dx = (abs(plus-minus)/(plus+minus)*100) if (plus+minus) else 0.0
    dx_list = [None]*length; dx_list[n] = dx
    adx_list = [None]*length; adx_list[2*n-1] = sum(dx_list[n:2*n])/n
    for j in range(2*n, length):
        prev = adx_list[j-1] or adx_list[2*n-1]
        adx_list[j] = ((prev*(n-1)) + dx_list[j]) / n
    return adx_list[-1]

def cci(high: List[float], low: List[float], close: List[float], period: int = 20) -> Optional[float]:
    if not close or len(close) < period:
        return None
    tp = [(high[i]+low[i]+close[i])/3 for i in range(len(close))]
    sma = sum(tp[-period:])/period
    md  = sum(abs(x-sma) for x in tp[-period:]) / period
    if md == 0: return 0.0
    return (tp[-1]-sma)/(0.015*md)

def williams_r(high: List[float], low: List[float], close: List[float], period: int = 14) -> Optional[float]:
    if not close or len(close) < period:
        return None
    hh = max(high[-period:]); ll = min(low[-period:])
    if hh == ll: return 0.0
    return (hh-close[-1])/(hh-ll)*-100

# Dummy class so other modules that do “from modules.technical_indicators import TechnicalIndicators” still work:
class TechnicalIndicators:
    pass

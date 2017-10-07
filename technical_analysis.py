import talib
import numpy as np


class TechnicalAnalysis:
    def __init__(self, open, high, low, close, volume):
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.average_price = (high - low) / 2
        self.volume = volume

        self.setter_periods()

    def setter_periods(self, ema_f_period=6, ema_m_period=12, ema_s_period=24):
        self.ema_f_period = ema_f_period
        self.ema_m_period = ema_m_period
        self.ema_s_period = ema_s_period

    def execute(self):
        ema_f = talib.EMA(self.close, self.ema_f_period)
        ema_m = talib.EMA(self.close, self.ema_m_period)
        ema_s = talib.EMA(self.close, self.ema_s_period)

        spread_ema_f = ema_f - self.close
        spread_ema_m = ema_m - self.close
        spread_ema_s = ema_s - self.close

        return np.array([self.close, spread_ema_f, spread_ema_m, spread_ema_s]).transpose()

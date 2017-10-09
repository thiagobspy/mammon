import talib
import numpy as np


class TechnicalAnalysis:
    def __init__(self, open, high, low, close, volume):
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

        self.setter_periods()

    def setter_periods(self, ema_period=14,
                       rsi_period=14,
                       stoch_period=14,
                       mom_period=14,
                       adx_period=14,
                       willr_period=14,
                       cci_period=14,
                       roc_period=10,
                       stochrsi_period=14,
                       trix_period=30,
                       mfi_period=14,
                       ultosc_1_period=7,
                       ultosc_2_period=14,
                       ultosc_3_period=21,
                       aroon_period=14,
                       aroonosc_period=14,
                       atr_period=14,
                       adoscfast_period=3,
                       adoscslow_period=10):

        self.ema_period = ema_period
        self.rsi_period = rsi_period
        self.stoch_period = stoch_period
        self.mom_period = mom_period
        self.adx_period = adx_period
        self.willr_period = willr_period
        self.cci_period = cci_period
        self.roc_period = roc_period
        self.stochrsi_period = stochrsi_period
        self.trix_period = trix_period
        self.mfi_period = mfi_period
        self.ultosc_1_period = ultosc_1_period
        self.ultosc_2_period = ultosc_2_period
        self.ultosc_3_period = ultosc_3_period
        self.aroon_period = aroon_period
        self.aroonosc_period = aroonosc_period
        self.atr_period = atr_period
        self.adoscfast_period = adoscfast_period
        self.adoscslow_period = adoscslow_period

    def execute(self):
        ema = talib.EMA(self.close, self.ema_period)
        rsi = talib.RSI(self.close, self.rsi_period)
        stoch_K, stoch_D = talib.STOCHF(self.high, self.low, self.close, fastk_period=self.stoch_period)
        macd, macdsignal, macdhist = talib.MACD(self.close)
        sar = talib.SAR(self.high, self.low)
        mom = talib.MOM(self.close, self.mom_period)
        adx = talib.ADX(self.high, self.low, self.close, self.adx_period)
        willr = talib.WILLR(self.high, self.low, self.close, self.willr_period)
        cci = talib.CCI(self.high, self.low, self.close, self.cci_period)
        roc = talib.ROC(self.close, self.roc_period)
        stochrsi_K, stochrsi_D = talib.STOCHRSI(self.close, self.stochrsi_period)
        trix = talib.TRIX(self.close, self.trix_period)
        mfi = talib.MFI(self.high, self.low, self.close, self.volume, self.mfi_period)
        ultosc = talib.ULTOSC(self.high, self.low, self.close, self.ultosc_1_period, self.ultosc_2_period, self
                              .ultosc_3_period)
        aroon_down, aroon_up = talib.AROON(self.high, self.low, self.aroon_period)
        aroonosc = talib.AROONOSC(self.high, self.low, self.aroonosc_period)
        atr = talib.ATR(self.high, self.low, self.close, self.atr_period)
        ad = talib.AD(self.high, self.low, self.close, self.volume)
        obv = talib.OBV(self.close, self.volume)
        adosc = talib.ADOSC(self.high, self.low, self.close, self.volume, self.adoscfast_period, self.adoscslow_period)

        spread_ema = ema - self.close
        spread_stoch = stoch_K - stoch_D
        spread_stochrsi = stochrsi_K - stochrsi_D
        spread_aroon = aroon_up - aroon_down

        return np.array([self.close, spread_ema, rsi, macd, macdhist, stoch_K, spread_stoch, sar, mom, adx,
                         willr, cci, roc, spread_stochrsi, trix, mfi, ultosc, aroon_down, aroon_up,
                         spread_aroon, aroonosc, atr, ad, obv, adosc]).transpose()
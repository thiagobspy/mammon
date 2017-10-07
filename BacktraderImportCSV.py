import itertools
from datetime import datetime

import backtrader as bt
from backtrader import date2num
from mt_all_stats.technical_analysis import TechnicalAnalysis
from mt_all_stats.utils import *

from neural_network import NeuralNetwork


class MammonCSV(bt.CSVDataBase):
    def _loadline(self, linetokens):
        i = itertools.count(0)

        dttxt = linetokens[next(i)]
        dt = bt.date(int(dttxt[0:4]), int(dttxt[5:7]), int(dttxt[8:10]))
        dtnum = date2num(datetime.combine(dt, self.p.sessionend))

        self.lines.datetime[0] = dtnum
        o = float(linetokens[next(i)])
        h = float(linetokens[next(i)])
        l = float(linetokens[next(i)])
        c = float(linetokens[next(i)])
        v = float(linetokens[next(i)])

        self.lines.open[0] = o
        self.lines.high[0] = h
        self.lines.low[0] = l
        self.lines.close[0] = c
        self.lines.volume[0] = v

        return True


class MammonStrategy(bt.Strategy):
    def start(self):
        self.bar = 0
        self.last_target = [1, 1]
        self.order = None
        self.max = np.array([1.08950000e-02, 1.75504860e-03, 9.51463870e+01,
                             1.07903849e-03, 7.24786276e-04, 1.00000000e+02,
                             6.56716418e+01, 1.39936000e+00, 9.65500000e-03,
                             8.07219885e+01, -0.00000000e+00, 4.66666667e+02,
                             1.73000000e+04, 6.66666667e+01, 8.17678671e+00,
                             1.00000000e+02, 8.97534003e+01, 1.00000000e+02,
                             1.00000000e+02, 1.00000000e+02, 1.00000000e+02,
                             4.50666654e-03, 1.63476617e+04, 3.56987300e+06,
                             4.08622160e+03])
        self.min = np.array([0.00000000e+00, -8.47299988e-03, 3.83920267e+01,
                             -4.05831472e-04, -3.40956987e-04, 0.00000000e+00,
                             -6.41975309e+01, 1.03406000e+00, -1.02350000e-02,
                             5.06191443e+00, -1.00000000e+02, -4.66666667e+02,
                             -1.00000000e+02, -6.66666667e+01, -2.42719530e+00,
                             -2.92706736e-11, 1.07646703e+01, 0.00000000e+00,
                             0.00000000e+00, -1.00000000e+02, -1.00000000e+02,
                             5.77056576e-05, -5.40155216e+05, -1.00800000e+03,
                             -3.72679717e+03])

        self.neural_network = NeuralNetwork()
        self.neural_network.load_weight('eur_usd_m5_weight.txt')

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            if order.isbuy():
                self.log( 'BUY EXECUTED, Cost: %.5f,  PNL: %.5f' % (order.executed.value, order.executed.pnl))
            else:  # Sell
                self.log( 'SELL EXECUTED, Cost: %.5f,  PNL: %.5f' % (order.executed.value, order.executed.pnl))

            self.order = None

    def next(self):
        if self.bar < 118:
            self.bar += 1
            return

        if self.order:
            return

        open = np.array([self.data.open[x] for x in range(-117, 1)])
        high = np.array([self.data.high[x] for x in range(-117, 1)])
        low = np.array([self.data.low[x] for x in range(-117, 1)])
        close = np.array([self.data.close[x] for x in range(-117, 1)])
        volume = np.array([self.data.volume[x] for x in range(-117, 1)])

        data_factory = TechnicalAnalysis(open, high, low, close, volume)
        data = data_factory.execute()
        data = Utils.remove_nan(data)
        data = Utils.normalize(data, self.max, self.min)

        target = self.neural_network.predict(np.array([data]))
        actual_target = target.argmax()

        if actual_target == 0 or actual_target == 2:
            if self.last_target[0] != actual_target:
                self.last_target[0] = actual_target
            elif self.last_target[1] != actual_target:
                self.last_target[1] = actual_target
            else:
                self.last_target = [1, 1]
                if actual_target == 0:
                    self.order = self.buy(exectype=bt.Order.StopTrail, trailpercent=0.001)
                else:
                    self.order = self.sell(exectype=bt.Order.StopTrail, trailpercent=0.001)


def runstrategy():
    cerebro = bt.Cerebro()

    data = MammonCSV(dataname='eur_usd_m5.csv')

    cerebro.adddata(data)

    cerebro.addstrategy(MammonStrategy)

    cerebro.broker.set_cash(10000)

    cerebro.run()
    print(cerebro.broker.cash)


runstrategy()

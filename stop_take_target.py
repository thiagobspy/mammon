import numpy as np


class StopTakeTarget:
    def __init__(self, high, low, open):
        self.high = high
        self.low = low
        self.open = open

        self.setter_params()

    def setter_params(self, spread_up=0.001, spread_down=0.001, limit_bar=30):
        self.spread_up = spread_up
        self.spread_down = spread_down
        self.limit_bar = limit_bar

    def execute(self):
        targets = []
        for count in range(len(self.open)):
            up_target = self.open[count] * (1 + self.spread_up)
            down_target = self.open[count] * (1 - self.spread_down)
            actual_bar = count

            target = np.array([0, 1, 0])

            while (actual_bar + self.limit_bar < len(self.open)):

                if self.high[actual_bar] > up_target:
                    target = np.array([1, 0, 0])
                    break
                elif self.low[actual_bar] < down_target:
                    target = np.array([0, 0, 1])
                    break
                elif actual_bar >= count + self.limit_bar:
                    break

                actual_bar += 1

            targets.append(target)
        return np.array(targets)


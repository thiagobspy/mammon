from game import Game
from technical_analysis import TechnicalAnalysis


class Forex(Game):
    def __init__(self, data, start_bar=300, take_profit=0.001, stop_loss=0.001, times_serie=60):
        self.data = data
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.times_serie = times_serie
        self.start_bar = start_bar
        self.last_bar = data.shape[0]

        self.split_data()
        self.reset()

    @property
    def name(self):
        return 'Forex'

    @property
    def nb_actions(self):
        return 3

    # open, high, low, close, volume
    def split_data(self):
        self.open = self.data[:, 1]
        self.high = self.data[:, 2]
        self.low = self.data[:, 3]
        self.close = self.data[:, 4]
        self.volume = self.data[:, 5]
        technical_analysis = TechnicalAnalysis(self.open, self.high, self.low, self.close, self.volume)
        self.technical_analysis = technical_analysis.execute()

    def reset(self):
        self.actual_bar = self.start_bar
        self.profit = 0

    def play(self, action):
        assert action in range(self.nb_actions), 'Invalid action.'
        profit_actual_bar = self.execute_order(action)
        self.profit += profit_actual_bar

    def execute_order(self, action):
        if action == 1:
            profit = self.buy_position()
        elif action == 2:
            profit = self.sell_position()
        else:
            profit = self.neutral_position()
        return profit

    def buy_position(self):
        price = self.close[self.actual_bar]
        target_price = price * (1 + self.take_profit)
        stop_price = price * (1 - self.stop_loss)
        self.actual_bar += 1

        while self.last_bar > self.actual_bar:
            high = self.high[self.actual_bar]
            low = self.low[self.actual_bar]
            if high > target_price:
                return target_price - price
            if low < stop_price:
                return stop_price - price
            self.actual_bar += 1
        self.reset()
        return 0

    def sell_position(self):
        price = self.close[self.actual_bar]
        target_price = price * (1 - self.take_profit)
        stop_price = price * (1 + self.stop_loss)
        self.actual_bar += 1

        while self.last_bar > self.actual_bar:
            high = self.high[self.actual_bar]
            low = self.low[self.actual_bar]
            if high > stop_price:
                return price - stop_price
            if low < target_price:
                return price - target_price
            self.actual_bar += 1
        self.reset()
        return 0

    def neutral_position(self):
        self.actual_bar += 1
        return 0

    def get_state(self):
        start = self.actual_bar - self.times_serie
        end = self.actual_bar
        return self.technical_analysis[start:end, :]

    def get_score(self):
        return self.profit

    def is_over(self):
        return self.profit < 0

    def is_won(self):
        return self.profit >= 0

    def get_actual_bar(self):
        return self.actual_bar

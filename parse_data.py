import numpy as np
import pandas as pd


class ParseData:
    def __init__(self, file, delimiter=','):
        self.file = file
        self.delimiter = delimiter

    # open, high, low, close, volume
    def parse(self):
        data = np.genfromtxt(self.file, delimiter=self.delimiter)
        return data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5]

    def get_file(self):
        data = np.genfromtxt(self.file, delimiter=self.delimiter)
        return data

    def get_data_per_week(self):
        df = pd.read_csv(self.file, parse_dates=['date'])
        df['date'] = pd.to_datetime(df['date'])

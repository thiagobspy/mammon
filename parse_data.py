from datetime import timedelta

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

        min_date = df['date'].min()
        max_date = df['date'].max()

        first_day_week = self.setting_date_for_monday_midnight(min_date)
        weeks = []

        while first_day_week < max_date:
            last_day_week = first_day_week + timedelta(days=6)
            week_data = df[(df['date'] >= first_day_week) & (df['date'] < last_day_week)]
            data_without_date = week_data.values[:,1:]
            data = data_without_date.astype(float)
            weeks.append(data)
            first_day_week += timedelta(days=7)

        return weeks

    def setting_date_for_monday_midnight(self, date):
        monday_first = date - timedelta(days=date.weekday())
        return monday_first.replace(hour=0, minute=0, second=0)

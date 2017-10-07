import numpy as np


class Utils:
    @staticmethod
    def normalize(data, max, min):
        return (data - min) / (max - min)

    @staticmethod
    def remove_nan(data):
        return data[~np.isnan(data).any(axis=1)]

    @staticmethod
    def concatenate_columns(input_data, target_data):
        return np.append(input_data, target_data, axis=1)

    @staticmethod
    def concatenate_line(data_one, data_two):
        return np.append(data_one, data_two, axis=0)

    @staticmethod
    def split_recurrent_data_per_times_series(input_data, target_data, times_series):
        input_data = np.flip(input_data, axis=0)
        target_data = np.flip(target_data, axis=0)

        features = []
        features_target = []
        for count in range(len(input_data) - times_series):
            features.append(input_data[count: count + times_series])
            features_target.append(target_data[count])
        return np.array(features), np.array(features_target)

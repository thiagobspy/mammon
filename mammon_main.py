import utils
from parse_data import ParseData
from stop_take_target import StopTakeTarget
from technical_analysis import TechnicalAnalysis
from utils import *

from neural_network import NeuralNetwork

parse_data = ParseData('eur_usd_m5.csv')
data_per_week = parse_data.get_data_per_week()
open, high, low, close, volume = Utils.split_arrays(data_per_week[3], 5)

technical_analysis = TechnicalAnalysis(open, high, low, close, volume)
input_data = technical_analysis.execute()
number_features = input_data.shape[1]

stop_take_target = StopTakeTarget(high, low, open)
target_data = stop_take_target.execute()

times_series = 30
data = Utils.concatenate_columns(input_data, target_data)
data = Utils.remove_nan(data)
data = Utils.normalize_zero_one(data, data.max(axis=0), data.min(axis=0))
input_data_treated, target_data_treated = Utils.split_recurrent_data_per_times_series(data[:, 0:number_features],
                                                                                      data[:, number_features:],
                                                                                      times_series)
print(input_data_treated.shape)
print(target_data_treated.shape)
print(target_data.sum(axis=0))

input_shape = input_data_treated.shape[1:3]
output_layer = target_data_treated.shape[1]

neural_network = NeuralNetwork()
neural_network.create_model(input_shape=input_shape, hidden_layer=128, output_layer=output_layer)
neural_network.compile_model()
neural_network.add_data(input_data_treated, target_data_treated)
neural_network.fit(validation_split=0.2, batch_size=32, epochs=100, verbose=1)
neural_network.save_model('eur_usd_m5_model.txt')
neural_network.save_weight('eur_usd_m5_weight.txt')

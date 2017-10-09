from parse_data import ParseData
from stop_take_target import StopTakeTarget
from technical_analysis import TechnicalAnalysis
from utils import *

from neural_network import NeuralNetwork

parse_data = ParseData('eur_usd_m5.csv')
open, high, low, close, volume = parse_data.parse()

technical_analysis = TechnicalAnalysis(open, high, low, close, volume)
input_data = technical_analysis.execute()
number_features = input_data.shape[1]

stop_take_target = StopTakeTarget(high, low, open)
target_data = stop_take_target.execute()

times_series = 90
data = Utils.concatenate_columns(input_data, target_data)
data = Utils.remove_nan(data)
data = Utils.normalize_minus_one_more_one(data, data.__abs__().max(axis=0))
input_data_treated, target_data_treated = Utils.split_recurrent_data_per_times_series(data[:, 0:number_features],
                                                                                      data[:, number_features:],
                                                                                      times_series)
print(target_data.sum(axis=0))
print(input_data_treated.shape)
print(target_data_treated.shape)

input_shape = input_data_treated.shape[1:3]
output_layer = target_data_treated.shape[1]

neural_network = NeuralNetwork()
neural_network.create_model(input_shape=input_shape, output_layer=output_layer)
neural_network.compile_model()
neural_network.add_data(input_data_treated, target_data_treated)
neural_network.fit(validation_split=0.2, batch_size=128, epochs=10)
neural_network.save_model('eur_usd_m5_model.txt')
neural_network.save_weight('eur_usd_m5_weight.txt')

import matplotlib.pyplot as plt
from keras.models import model_from_json
from sklearn.model_selection import train_test_split

from neural_network import NeuralNetwork
from parse_data import ParseData
from stop_take_target import StopTakeTarget
from technical_analysis_complete import TechnicalAnalysisComplete
from utils import *


def prepare_data(open, high, low, close, volume):
    technical_analysis = TechnicalAnalysisComplete(open, high, low, close, volume)
    input_data = technical_analysis.execute()
    number_features = input_data.shape[1]

    stop_take_target = StopTakeTarget(high, low, open)
    target_data = stop_take_target.execute()

    times_series = 5
    data = Utils.concatenate_columns(input_data, target_data)
    data = Utils.remove_nan(data)
    input_data_treated, target_data_treated = Utils.split_recurrent_data_per_times_series(data[:, 0:number_features],
                                                                                          data[:, number_features:],
                                                                                          times_series)
    return input_data_treated, target_data_treated


parse_data = ParseData('info/eur_usd_m15_.csv')
open_t, high, low, close, volume = parse_data.parse()
input_data, target_data = prepare_data(open_t, high, low, close, volume)

print(input_data.shape)
print(target_data.shape)

model_json = open('info/model_(5,141)_m15.json').read()
model = model_from_json(model_json)
model.load_weights('info/weights_(5,141)_m15.h5')

predict = model.predict(input_data)

total, acerto = 1, 0
for pred, targ in zip(predict, target_data):
    if (pred[pred > 0.9].shape[0]):
        total += 1
        acerto += pred.argmax() == targ.argmax()
acc = acerto / total
print('Total: ', total)
print('Acerto: ', acerto)
print('Acc: ', acc)

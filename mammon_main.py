import matplotlib.pyplot as plt
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


def execute(input_data_treated, target_data_treated):
    print(input_data_treated.shape)
    print(target_data_treated.shape)
    print(target_data_treated.sum(axis=0))

    input_shape = input_data_treated.shape[1:3]
    output_layer = target_data_treated.shape[1]

    neural_network = NeuralNetwork()
    neural_network.create_model(input_shape=input_shape, hidden_layer=64, output_layer=output_layer)
    neural_network.compile_model()
    neural_network.add_data(input_data_treated, target_data_treated)
    return neural_network


parse_data = ParseData('info/eur_usd_m15.csv')
# open_t, high, low, close, volume = parse_data.parse()
data_per_week = parse_data.get_data_per_week()
data_per_week = data_per_week[:-10]
data_per_week_t = data_per_week[-10:]

open_t, high, low, close, volume = Utils.split_arrays(data_per_week[0], 5)
input_data, target_data = prepare_data(open_t, high, low, close, volume)

for week in range(1, len(data_per_week)):
    open_t, high, low, close, volume = Utils.split_arrays(data_per_week[week], 5)
    X, Y = prepare_data(open_t, high, low, close, volume)
    input_data = np.concatenate((input_data, X), axis=0)
    target_data = np.concatenate((target_data, Y), axis=0)

X_train, X_test, Y_train, Y_test = train_test_split(input_data, target_data, test_size=0.1, random_state=42)

print('\nShape: ', X_train.shape)
nn = execute(X_train, Y_train)
model = nn.get_model()
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), verbose=1, epochs=20)

model.save_weights('info/weights_(5,141)_m15.h5')
with open('info/model_(5,141)_m15.json', 'w') as file:
    file.write(model.to_json())

open_t, high, low, close, volume = Utils.split_arrays(data_per_week_t[0], 5)
input_data, target_data = prepare_data(open_t, high, low, close, volume)

for week in range(1, len(data_per_week_t)):
    open_t, high, low, close, volume = Utils.split_arrays(data_per_week_t[week], 5)
    X, Y = prepare_data(open_t, high, low, close, volume)
    input_data = np.concatenate((input_data, X), axis=0)
    target_data = np.concatenate((target_data, Y), axis=0)

X_train = input_data
Y_train = target_data

predict_data = model.predict(X_train)
total, acerto = 1, 0
for pred, targ in zip(predict_data, Y_train):
    if (pred[pred > 0.9].shape[0]):
        total += 1
        acerto += pred.argmax() == targ.argmax()
acc = acerto / total
print('Total: ', total)
print('Acerto: ', acerto)
print('Acc: ', acc)

"""
epoch_train = 10
confidence = [0.3, 0.5, 0.7, 0.9]
history = {confidence[0]: [],
           confidence[1]: [],
           confidence[2]: [],
           confidence[3]: []}

history_t = {confidence[0]: [],
           confidence[1]: [],
           confidence[2]: [],
           confidence[3]: []}

history_a = {confidence[0]: [],
           confidence[1]: [],
           confidence[2]: [],
           confidence[3]: []}

for conf in confidence:
    neural_network = execute(X_train, Y_train)
    for i in range(0, epoch_train):
        neural_network.fit(batch_size=64, epochs=1, verbose=0)
        predict_data = neural_network.predict(X_test)
        total, acerto = 1, 0
        for pred, targ in zip(predict_data, Y_test):
            if (pred[pred > conf].shape[0]):
                total += 1
                acerto += pred.argmax() == targ.argmax()
        acc = acerto / total
        print('Acerto (', conf, '):', acc)
        history[conf].append(acc)
        history_t[conf].append(total)
        history_a[conf].append(acerto)

plt.plot(history_t[0.3])
plt.plot(history_t[0.5])
plt.plot(history_t[0.7])
plt.plot(history_t[0.9])
plt.title('Total de operações')
plt.ylabel('Operações')
plt.xlabel('epoch')
plt.legend(['Sem confiança', '50% confiança', '70% confiança', '90% confiança'], loc='upper left')
plt.show()

plt.plot(history_a[0.3])
plt.plot(history_a[0.5])
plt.plot(history_a[0.7])
plt.plot(history_a[0.9])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Sem confiança', '50% confiança', '70% confiança', '90% confiança'], loc='upper left')
plt.show()

plt.plot(history[0.3])
plt.plot(history[0.5])
plt.plot(history[0.7])
plt.plot(history[0.9])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['34%', '50%', '70%', '90%'], loc='upper right')
plt.show()
"""

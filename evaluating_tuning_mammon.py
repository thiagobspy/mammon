import numpy
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import BatchNormalization, LSTM, Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from parse_data import ParseData
from stop_take_target import StopTakeTarget
from technical_analysis_complete import TechnicalAnalysisComplete
from utils import Utils


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


def build_classifier(optimizer, init, hidden_neuron, hidden_layer):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(5, 141)))
    model.add(LSTM(hidden_neuron, kernel_initializer=init, dropout=0.3, recurrent_dropout=0.3))
    for i in range(hidden_layer):
        model.add(Dense(hidden_neuron * 2, kernel_initializer=init, activation='relu'))
        model.add(Dropout(0.3))
    model.add(Dense(3, kernel_initializer=init, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


parse_data = ParseData('info/eur_usd_m15.csv')
data_per_week = parse_data.get_data_per_week()

open_d, high, low, close, volume = Utils.split_arrays(data_per_week[0], 5)
input_data, target_data = prepare_data(open_d, high, low, close, volume)

for week in range(1, len(data_per_week)):
    open_d, high, low, close, volume = Utils.split_arrays(data_per_week[week], 5)
    X, Y = prepare_data(open_d, high, low, close, volume)
    input_data = numpy.concatenate((input_data, X), axis=0)
    target_data = numpy.concatenate((target_data, Y), axis=0)

model = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size': [64, 1024],
              'epochs': [50],
              'init': ['glorot_uniform', 'normal', 'uniform'],
              'optimizer': ['adam', 'rmsprop'],
              'hidden_neuron': [32, 64, 128],
              'hidden_layer': [0, 1]}

early = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, verbose=0, mode='auto')
checkpointer = ModelCheckpoint(filepath='info/weights.hdf5', verbose=0, save_best_only=True)
callbacks_list = [early, checkpointer]

grid_search = GridSearchCV(estimator=model, param_grid=parameters, cv=2, n_jobs=4, verbose=0)
grid_result = grid_search.fit(input_data, target_data, verbose=2, validation_split=0.2, callbacks=callbacks_list)

line = "Best: %f using %s\n" % (grid_result.best_score_, grid_result.best_params_)

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
best_model = grid_result.best_estimator_.model

for mean, stdev, param in zip(means, stds, params):
    line += "%f (%f) with: %r\n" % (mean, stdev, param)

with open('info/grid_result_(5,141)_m15', 'w') as file:
    file.write(line)
with open('info/model_(5,141)_m15.json', 'w') as file:
    file.write(best_model.to_json())

best_model.save_weights('info/weights_(5,141)_m15.h5')



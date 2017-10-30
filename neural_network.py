from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop, Adam, rmsprop
from keras import regularizers


class NeuralNetwork:
    def create_model(self, input_shape=(60, 25), hidden_layer=64, output_layer=3):
        self.model = Sequential()
        self.model.add(BatchNormalization(input_shape=input_shape))
        self.model.add(LSTM(hidden_layer, dropout=0.3, recurrent_dropout=0.3, return_sequences=False))
        self.model.add(Dense(hidden_layer, activation='relu'))
        self.model.add(Dense(output_layer, activation='softmax'))

    def compile_model(self, learning_rate=0.001, decay=1e-6):
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=rmsprop(lr=learning_rate, decay=decay),
                           metrics=['accuracy'])

    def fit(self, validation_split=0.00, batch_size=64, epochs=30, verbose=1):
        self.model.fit(self.input_data, self.output_data, validation_split=validation_split, batch_size=batch_size,
                       verbose=verbose, epochs=epochs)

    def predict(self, data):
        return self.model.predict(data, verbose=0)

    def save_model(self, filepath):
        self.model.save(filepath=filepath)

    def save_weight(self, filepath):
        self.model.save_weights(filepath=filepath)

    def load_weight(self, filepath):
        self.model.load_weights(filepath)

    def add_data(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data

    def get_model(self):
        return self.model

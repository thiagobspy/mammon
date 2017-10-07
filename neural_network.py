from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop, Adam
from keras import regularizers


class NeuralNetwork:
    def create_model(self, input_shape=(60, 25), hidden_layer=64, output_layer=3):
        self.model = Sequential()
        self.model.add(LSTM(hidden_layer,
                            input_shape=input_shape,
                            kernel_regularizer=regularizers.l2(0.01),
                            return_sequences=True))
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(hidden_layer, kernel_regularizer=regularizers.l2(0.01)))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(hidden_layer, kernel_regularizer=regularizers.l2(0.01), activation='tanh'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(output_layer, activation='sigmoid'))

    def compile_model(self, learing_rate=0.03, decay=1e-6, momentum=0.9, nesterov=True):
        sgd = Adam(lr=learing_rate, decay=decay)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    def fit(self, validation_split=0.20, batch_size=256, epochs=5):
        self.model.fit(self.input_data, self.output_data, validation_split=validation_split, batch_size=batch_size,
                       epochs=epochs)

    def predict(self, data):
        return self.model.predict(data)

    def save_model(self, filepath):
        self.model.save(filepath=filepath)

    def save_weight(self, filepath):
        self.model.save_weights(filepath=filepath)

    def load_weight(self, filepath):
        self.model.load_weights(filepath)

    def add_data(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data

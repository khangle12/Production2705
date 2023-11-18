import os
import numpy as np
import pandas as pd

from keras.models import Sequential
from tensorflow.keras import optimizers
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import classification_report


class MindWaveClassify:
    def __init__(self, n_class=3, epochs=10, batch_size=10, pre_trained_path=None):
        self.model = None
        self.n_epochs = epochs
        self.n_class = n_class
        self.batch_size = batch_size
        self.model_path = 'classify.h5'
        self.pre_trained_path = pre_trained_path
        self.data_path = 'data_characters_classification.csv'

    def build_model(self, input_dim):
        model = Sequential()
        model.add(LSTM(256, return_sequences=True, input_shape=input_dim))
        model.add(Dropout(0.5))
        model.add(LSTM(128))
        model.add(Dense(self.n_class, activation='softmax'))
        optimizer = optimizers.RMSprop(learning_rate=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def train(self, X, y):
        self.model = self.build_model(input_dim=(X.shape[1], X.shape[2]))
        print(self.model.summary())
        if os.path.exists(self.pre_trained_path):
            self.model.load_weights(self.pre_trained_path)
        self.model.fit(X, y, batch_size=self.batch_size, epochs=self.n_epochs, validation_split=0.3, shuffle=True)
        self.model.save_weights(self.model_path)

    def load_data(self):
        df = pd.read_csv(self.data_path, encoding='utf-8', header=None)
        df = df.to_numpy()
        np.random.shuffle(df)
        X = df[:, :-1]
        y = df[:, -1]
        y = np.eye(self.n_class)[y.astype(int)]
        X = np.reshape(X, (-1, 1, X.shape[1]))
        return X, y

    def load_model(self, input_dim):
        print('Loading model...')
        self.model = self.build_model(input_dim)
        print(self.model.summary())
        self.model.load_weights(self.model_path)

    def predict(self, X):
        if X.ndim == 1:
            X = np.reshape(X, (-1, 1, X.shape[0]))
        if self.model is None:
            self.load_model(input_dim=(X.shape[1], X.shape[2]))
        y = self.model.predict(X)
        return y

    def evaluate(self, X, y):
        if self.model is None:
            self.load_model(input_dim=(X.shape[1], X.shape[2]))
        y_pred = self.model.predict(X)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y, axis=1)
        print(classification_report(y_true, y_pred))

    def run(self):
        X, y = self.load_data()
        self.train(X, y)
        self.evaluate(X, y)


if __name__ == '__main__':
    classify = MindWaveClassify(n_class=3, epochs=10, batch_size=20, pre_trained_path = 'classify_1.h5')
    classify.run()

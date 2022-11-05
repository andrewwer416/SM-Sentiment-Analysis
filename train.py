import tensorflow as tf
import matplotlib as plt
from tensorflow import keras
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras.layers import Dense, Dropout, LSTM, CuDNNGRU, CuDNNLSTM, BatchNormalization, Bidirectional
import numpy as np
DROPOUT_RATE = 0.2
def train(X_train, X_val, y_train, y_val):
    model = Sequential()
    model.add(CuDNNLSTM(128, input_shape=(X_train.shape[1], 1), return_sequences=True))
    model.add(Dropout(DROPOUT_RATE))
    model.add(BatchNormalization())

    model.add(CuDNNLSTM(128, return_sequences=True))
    model.add(Dropout(DROPOUT_RATE))
    model.add(BatchNormalization())

    model.add(CuDNNLSTM(128, return_sequences=False))
    model.add(Dropout(DROPOUT_RATE))
    model.add(BatchNormalization())

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(DROPOUT_RATE))

    model.add(Dense(2, activation='softmax'))

    opt = Adam(learning_rate=0.001, decay=1e-6)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    history = model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_val, y_val), verbose=1)
    plot_results(history)

def plot_results(history):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), )

    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_ylabel('Loss')

    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_ylabel('Accuracy')

    plt.legend()
    plt.tight_layout()
    plt.show()

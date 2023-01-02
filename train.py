import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras.layers import Dense, Dropout, LSTM, BatchNormalization, GlobalAveragePooling1D, Embedding, Bidirectional
import numpy as np
DROPOUT_RATE = 0.4
embedding_dim = 150
vocab_size = 40000
max_length = 20
batch_size = 512
def train(X_train, X_val, y_train, y_val):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
    # model.add(GlobalAveragePooling1D())
    model.add(Bidirectional(LSTM(64, input_shape=(X_train.shape[1], 1), return_sequences=True)))
    model.add(Dropout(DROPOUT_RATE))
    #model.add(BatchNormalization())

    #model.add(Bidirectional(LSTM(64, return_sequences=True)))
    #model.add(Dropout(DROPOUT_RATE))
    #model.add(BatchNormalization())

    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(DROPOUT_RATE))
    #model.add(BatchNormalization())

    model.add(Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2()))
    #model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(1, activation='sigmoid'))
    opt = RMSprop()
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=5, validation_data=(X_val, y_val), verbose=1, callbacks=callback)
    plot_results(history)
    model.summary()
    model.save("my_model")
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

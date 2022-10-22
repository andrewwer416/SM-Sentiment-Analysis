import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, CuDNNGRU, CuDNNLSTM, BatchNormalization, Bidirectional


def model:
    model = Sequential()
    model.add(Bidirectional(CuDNNLSTM(128))
    )
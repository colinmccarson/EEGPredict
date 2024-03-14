import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, LSTM, GRU, RNN, BatchNormalization, MaxPooling2D, Reshape
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
import torch.nn.functional as tnnf
import os


class CNN:
    def __init__(self, filters, kernel_size, dropout, l2_lambda, num_deep, num_fc):
        self.model = Sequential()
        # Use l2 reg on all weights
        # Somewhat smaller VGGnet, try to keep the filter size smaller
        # 2x conv - pool - sbn - relu - dropout
        self.model.add(Conv2D(filters=filters, kernel_size=kernel_size, padding='same', input_shape=(400, 1, 22),
                              kernel_regularizer=l2(l2_lambda), use_bias=True, activation='relu'))
        self.add_standard_conv2d(1, filters, kernel_size, l2_lambda)
        filters *= 2
        self.add_deep_conv2d(num_deep, filters, kernel_size, dropout, l2_lambda)
        self.model.add(Flatten())
        filters *= (2**num_deep)
        # 2x FC
        self.add_standard_fc(num_fc, filters, dropout, l2_lambda)

        # Output layer with Softmax activation
        self.model.add(Dense(4, activation='softmax'))  # Output FC layer with softmax activation

    def add_standard_conv2d(self, howmany, filters, kernel_size, l2_lambda):
        # Mutates model
        # conv - pool - sbn - relu - dropout
        for i in range(howmany):
            self.model.add(Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
                                  kernel_regularizer=l2(l2_lambda), use_bias=True))
            self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=2))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))
        return filters * 2

    def add_standard_fc(self, howmany, hidden_dim, dropout, l2_lambda):
        self.model.add(Dense(hidden_dim, activation='relu', kernel_regularizer=l2(l2_lambda)))
        self.model.add(Dropout(dropout))

    def add_deep_conv2d(self, howmany, filters, kernel_size, dropout, l2_lambda, depth=2, filters_double=True):
        for i in range(howmany):
            for j in range(depth):
                self.model.add(Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
                                      kernel_regularizer=l2(l2_lambda), use_bias=True, activation='relu'))
            self.model.add(BatchNormalization())
            self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=2))
            if filters_double:
                filters *= 2

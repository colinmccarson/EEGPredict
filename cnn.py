import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, LSTM, GRU, RNN, BatchNormalization, MaxPooling2D, Reshape, Conv3D, MaxPooling3D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
import torch.nn.functional as tnnf
import os


class VGG_INSPIRED_CNN:
    def __init__(self, filters, kernel_size, dropout, l2_lambda, num_deep, num_fc, use_batchnorm=True,
                 use_conv_dropout=False):
        self.archname = ('vgg_insp_' + "depth" + str(num_deep) + "_fc" + str(num_fc) + "_bn" + str(
            use_batchnorm) + '_f' + str(filters) + 'f k' + str(kernel_size)
                         + 'k d' + str(dropout) + 'd' + "_convdrop_" + str(use_conv_dropout)).replace('.', '_')
        self.use_batchnorm = use_batchnorm
        self.use_conv_dropout = use_conv_dropout
        for c in '()., ':
            self.archname = self.archname.replace(c, '_')
        self.model = Sequential()
        # Use l2 reg on all weights
        # VGG-ish
        self.model.add(Conv2D(filters=filters, kernel_size=kernel_size, padding='same', input_shape=(400, 1, 22),
                              kernel_regularizer=l2(l2_lambda), use_bias=True))
        if self.use_batchnorm:
            self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        if self.use_conv_dropout:
            self.model.add(Dropout(dropout))

        self.add_standard_conv(1, filters, kernel_size, dropout, l2_lambda)
        filters *= 2
        self.add_deep_conv(num_deep, filters, kernel_size, dropout, l2_lambda)
        self.model.add(Flatten())
        filters *= (2 ** num_deep)
        # 2x FC
        self.add_standard_fc(num_fc, filters, dropout, l2_lambda)

        # Output layer with Softmax activation
        self.model.add(Dense(4, activation='softmax'))  # Output FC layer with softmax activation

    def add_standard_conv(self, howmany, filters, kernel_size, dropout, l2_lambda):
        # Mutates model
        # conv - sbn - relu - dropout - pool
        for i in range(howmany):
            self.model.add(Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
                                  kernel_regularizer=l2(l2_lambda), use_bias=True))
            if self.use_batchnorm:
                self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))
            if self.use_conv_dropout:
                self.model.add(Dropout(dropout))
            self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=2))
        return filters * 2

    def add_standard_fc(self, howmany, hidden_dim, dropout, l2_lambda):
        self.model.add(Dense(hidden_dim, activation='relu', kernel_regularizer=l2(l2_lambda)))
        self.model.add(Dropout(dropout))

    def add_deep_conv(self, howmany, filters, kernel_size, dropout, l2_lambda, depth=2, filters_double=True):
        for i in range(howmany):
            for j in range(depth):
                self.model.add(Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
                                      kernel_regularizer=l2(l2_lambda), use_bias=True))
                if self.use_batchnorm:
                    self.model.add(BatchNormalization())
                self.model.add(Activation('relu'))
                if self.use_conv_dropout:
                    self.model.add(Dropout(dropout))
            self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=2))
            if filters_double:
                filters *= 2


class SimpleConv:
    def __init__(self, filters, kernel_size, dropout, l2_lambda):
        self.archname = ('simple_conv_' + str(filters) + str(kernel_size) + str(dropout)).replace('.', '_')
        for c in '()., ':
            self.archname = self.archname.replace(c, '_')
        self.model = Sequential()
        self.model.add(Conv2D(filters=filters, kernel_size=kernel_size, padding='same', input_shape=(400, 1, 22),
                              kernel_regularizer=l2(l2_lambda), use_bias=True, activation='relu'))
        self.model.add(Dropout(dropout))
        self.model.add(Conv2D(filters=filters, kernel_size=kernel_size))
        self.model.add()

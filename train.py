import os

os.environ['KERAS_BACKEND'] = 'torch'
import torch

print(torch.cuda.get_device_name(0))
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
import cnn
from keras.callbacks import ReduceLROnPlateau
import callbacks


def train_data_prep(X, y, sub_sample, average, noise):
    total_X = None
    total_y = None

    # Trimming the data (sample,22,1000) -> (sample,22,800)
    X = X[:, :, 0:800]
    print('Shape of X after trimming:', X.shape)

    # Maxpooling the data (sample,22,800) -> (sample,22,800/sub_sample)
    X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, sub_sample), axis=3)

    total_X = X_max
    total_y = y
    print('Shape of X after maxpooling:', total_X.shape)

    # Averaging + noise
    X_average = np.mean(X.reshape(X.shape[0], X.shape[1], -1, average), axis=3)
    X_average = X_average + np.random.normal(0.0, 0.5, X_average.shape)

    total_X = np.vstack((total_X, X_average))
    total_y = np.hstack((total_y, y))
    print('Shape of X after averaging + noise and concatenating:', total_X.shape)

    # Subsampling
    for i in range(sub_sample):
        X_subsample = X[:, :, i::sub_sample] + \
                      (np.random.normal(0.0, 0.5, X[:, :, i::sub_sample].shape) if noise else 0.0)

        total_X = np.vstack((total_X, X_subsample))
        total_y = np.hstack((total_y, y))

    print('Shape of X after subsampling and concatenating:', total_X.shape)
    print('Shape of Y:', total_y.shape)
    return total_X, total_y


def test_data_prep(X):
    total_X = None

    # Trimming the data (sample,22,1000) -> (sample,22,800)
    X = X[:, :, 0:800]
    print('Shape of X after trimming:', X.shape)

    # Maxpooling the data (sample,22,800) -> (sample,22,800/sub_sample)
    X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, 2), axis=3)

    total_X = X_max
    print('Shape of X after maxpooling:', total_X.shape)
    return total_X


## Loading and visualizing the data

if __name__ == '__main__':
    dpath = "./data/"
    X_test = np.load(dpath + "X_test.npy")
    y_test = np.load(dpath + "y_test.npy")
    person_train_valid = np.load(dpath + "person_train_valid.npy")
    X_train_valid = np.load(dpath + "X_train_valid.npy")
    print(X_train_valid.shape)
    y_train_valid = np.load(dpath + "y_train_valid.npy")
    person_test = np.load(dpath + "person_test.npy")

    ## Adjusting the labels so that

    # Cue onset left - 0
    # Cue onset right - 1
    # Cue onset foot - 2
    # Cue onset tongue - 3

    y_train_valid -= 769
    y_test -= 769

    # # # # #

    x_train, x_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.1)

    x_train, y_train = train_data_prep(x_train, y_train, 2, 2, True)
    x_valid, y_valid = train_data_prep(x_valid, y_valid, 2, 2, True)
    X_test_prep = test_data_prep(X_test)
    print('Shape of training set:', x_train.shape)
    print('Shape of validation set:', x_valid.shape)
    print('Shape of training labels:', y_train.shape)
    print('Shape of validation labels:', y_valid.shape)

    # Converting the labels to categorical variables for multiclass classification
    y_train = to_categorical(y_train, 4)
    y_valid = to_categorical(y_valid, 4)
    y_test = to_categorical(y_test, 4)
    print('Shape of training labels after categorical conversion:', y_train.shape)
    print('Shape of validation labels after categorical conversion:', y_valid.shape)
    print('Shape of test labels after categorical conversion:', y_test.shape)

    # Adding width of the segment to be 1
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], x_train.shape[2], 1)
    x_test = X_test_prep.reshape(X_test_prep.shape[0], X_test_prep.shape[1], X_test_prep.shape[2], 1)
    print('Shape of training set after adding width info:', x_train.shape)
    print('Shape of validation set after adding width info:', x_valid.shape)
    print('Shape of test set after adding width info:', x_test.shape)

    # Reshaping the training and validation dataset
    x_train = np.swapaxes(x_train, 1, 3)
    x_train = np.swapaxes(x_train, 1, 2)
    x_valid = np.swapaxes(x_valid, 1, 3)
    x_valid = np.swapaxes(x_valid, 1, 2)
    x_test = np.swapaxes(x_test, 1, 3)
    x_test = np.swapaxes(x_test, 1, 2)
    print('Shape of training set after dimension reshaping:', x_train.shape)
    print('Shape of validation set after dimension reshaping:', x_valid.shape)
    print('Shape of test set after dimension reshaping:', x_test.shape)

    # Compiling the model
    filters = 25
    kernel_size = (7, 7)
    dropout = .5
    l2_lambda = 0.001
    num_deep = 3
    num_fc = 2

    # Opt parameters
    learning_rate = 1e-3
    epochs = 25
    cnn_rnn_optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # Define early stopping criteria
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=1e-1, patience=2, min_lr=1e-6, mode='min')
    export_weights = callbacks.ExportModelWeights()
    # Add early stopping callback to the list of callbacks
    callbacks = [early_stopping, reduce_lr, export_weights]

    my_cnn = cnn.CNN(filters, kernel_size, dropout, l2_lambda, num_deep, num_fc)

    my_cnn.model.compile(loss='categorical_crossentropy',
                         optimizer=cnn_rnn_optimizer,
                         metrics=['accuracy'])

    # Training and validating the model
    cnn_rnn_model_results = my_cnn.model.fit(x_train,
                                             y_train,
                                             batch_size=64,
                                             epochs=epochs,
                                             validation_data=(x_valid, y_valid),
                                             callbacks=callbacks, verbose=True)

    ## Testing the hybrid CNN-RNN model
    cnn_rnn_score = my_cnn.model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy of the hybrid CNN-RNN model:', cnn_rnn_score[1])

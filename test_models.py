from keras.models import load_model
import os
import numpy as np
from keras.utils import to_categorical


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


X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

X_test_prep = test_data_prep(X_test)

y_test = to_categorical(y_test, 4)

x_test = X_test_prep.reshape(X_test_prep.shape[0], X_test_prep.shape[1], X_test_prep.shape[2], 1)

x_test = np.swapaxes(x_test, 1, 3)
x_test = np.swapaxes(x_test, 1, 2)


def find_keras_files(directory):
    keras_files = []
    for file in os.listdir(directory):
        if file.endswith(".keras"):
            keras_files.append(os.path.join(directory, file))
    return keras_files


directory = "./models_final"
keras_files = find_keras_files(directory)

for file in keras_files:
    model = load_model(file)
    model_score = model.evaluate(x_test, y_test, verbose=0)
    print(f'Test accuracy of {file}:', model_score[1])

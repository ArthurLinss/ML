
"""
use anaconda python 3.7 environment
https://elitedatascience.com/keras-tutorial-deep-learning-in-python

this example is about convolution (multi-layer) neural networks for image recognition with keras using the tensorflow backend
accuracy of the model should be around 99 %
"""

import numpy as np
np.random.seed(123)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt


def getData():
    """
    load and prepare the training and test data
    the data is already in keras (and shuffled)
    we return a dictinoary since we want to return more than just the data, e.g. also the number of distinct values of the targets
    """
    # load data and preprocess data (setting channels, setting data type, normalise)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print("training (X,y) shape: ", X_train.shape, y_train.shape)
    # declare channel (greyscale image -> 1, coloured would be 3)
    channels = 1
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], channels)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], channels)
    print("training (X,y) shape: ", X_train.shape, y_train.shape)
    # convert data type to float32
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    # normalize color values to unity
    X_train /= 255
    X_test /= 255

    # Convert 1-dimensional class arrays to 10-dimensional class matrices
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    print("training (X,y) shape: ", X_train.shape, Y_train.shape)

    n_unique_values_in_target = len(np.unique(y_test))

    data = {"data": ((X_train, Y_train), (X_test, Y_test)), "n_targets":n_unique_values_in_target}

    return data


def getConvNNModel(data):
    """
    define CNN
    """

    (X_train, y_train), (X_test, y_test) = data["data"]
    n_targets = data["n_targets"]

    model = Sequential()
    # input layer
    relu_activation = 'relu'
    softmax_activation = 'softmax'
    n_conv_filters = 32
    n_conv_kernels_row_col = (3,3)
    # add convolutional layer
    model.add(Convolution2D(n_conv_filters, n_conv_kernels_row_col, activation=relu_activation, input_shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3])))
    #print("model shape: ", model.output_shape)
    model.add(Convolution2D(n_conv_filters, n_conv_kernels_row_col, activation=relu_activation))
    # add pooling layer (downsampling)
    model.add(MaxPooling2D(pool_size=(2,2)))
    # dropout layer preventing overfittin via regularizing
    model.add(Dropout(0.25))
    # flatten -> make it 1-dimenstional
    model.add(Flatten())
    model.add(Dense(128, activation=relu_activation))
    model.add(Dropout(0.5))
    model.add(Dense(n_targets, activation=softmax_activation))

    # set loss function etc
    loss_function = 'categorical_crossentropy'
    optimizer = 'adam'
    metrics = ['accuracy']
    model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)

    return model


def fit(data, model):
    """
    fit the data using the model and get score
    """
    (X_train, y_train), (X_test, y_test) = data["data"]
    n_epochs = 2 # 10
    model.fit(X_train, y_train, batch_size=32, epochs=n_epochs, verbose=1)
    score = model.evaluate(X_test, y_test, verbose=0)
    print("score: ", score)


def keras_mnist():
    get_data = getData()
    model = getConvNNModel(get_data)
    fit(get_data, model)


def main():
    keras_mnist()

if __name__ == "__main__":
    main()

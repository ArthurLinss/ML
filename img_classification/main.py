from sklearn.datasets import fetch_openml
import sklearn
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

print("sklearn version: ", sklearn.__version__)


class prepareData():
    """
    class to load and prepare MNIST data for train and test
    """

    def __init__(self):
        self.dataSetName = "mnist_784"
        self.trainTestSplitVal = 60000
        self.getData()

    def getData(self):

        printInfo = False

        # Load data from https://www.openml.org/d/554
        X, y = fetch_openml(
            self.dataSetName, version=1, return_X_y=True, as_frame=False, parser="pandas"
        )

        if printInfo:
            print(X.shape)
            print(len(X))
            print(X)
            print(y.shape)
            print(y)


        # test train split
        trainTestSplitVal = self.trainTestSplitVal
        X_train, X_test, y_train, y_test = X[:trainTestSplitVal], X[trainTestSplitVal:], y[:trainTestSplitVal], y[trainTestSplitVal:]
        # shuffle train data (for x-val)
        shuffle_index = np.random.permutation(60000)
        X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test


    def plotDigit(self, data=None, label=None, number: int=36000, show: bool=False) -> None:
        if number < 0 or number > len(data):
            print("error in plotDigit: please provide number between 0 and %i" % len(data))
            return 1
        some_digit = data[number]
        some_digit_image = some_digit.reshape(28, 28)
        plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation="nearest")
        plt.title("Label: %s" % (label[number]))
        plt.axis("off")
        if show:
            plt.show()
        return 0

def main():
    dataClass = prepareData()
    X_train = dataClass.X_train
    y_train = dataClass.y_train
    X_test = dataClass.X_test
    y_test = dataClass.y_test

    #print(X_train, y_train)
    #dataClass.plotDigit(data=X_train, label=y_train, number=36000, show=True)



if __name__ == "__main__":
    t0 = time.time()
    main()
    run_time = time.time() - t0
    print("Run time in %.3f s" % run_time)

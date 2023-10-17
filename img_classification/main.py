"""
MNIST data set for classifying picutres of hand-written digits


accuracy in general not a good metric to measure performance in classifications (high accuaracy might be related to imbalanced data set)
precision = TP/(TP+FP)  (accuracy of positive predictions)
recall = Tp/(TP+FN)  (sensitivie, true positive rate)
"""


from sklearn.datasets import fetch_openml
import sklearn
import time
from sklearn.linear_model import SGDClassifier
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score

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


def xValOwnImpl(X_train, y_train, clf):
    """
    own implementation of cross valiation using strafified kfold
    """
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    skf.get_n_splits(X_train, y_train)
    for i, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")
        clone_clf = clone(clf)
        X_train_folds = X_train[train_index]
        y_train_folds = (y_train[train_index])
        X_test_fold = X_train[test_index]
        y_test_fold = (y_train[test_index])

        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_fold)
        n_correct = sum(y_pred == y_test_fold)
        print(n_correct / len(y_pred))

def binaryClass(X_train, y_train, targetNumber: str="5") -> None:
    """
    classification and corss-valdidation

    identify number vs. not-number
    sgd = stochastic gradient classifier
    targetNumber is a string here since we use fetch_openml() to download MNIST, and it returns labels as strings
    see: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
    """
    runXVal = False
    runConfusionMatrix = True

    # chose the value you want to classify, e.g. 5 to look for 5 vs. not 5
    y_train_label = (y_train == targetNumber)

    clf = SGDClassifier(random_state=42)
    clf.fit(X_train, y_train_label)

    #test
    #print(X_train[5000].reshape(1, -1))
    #print(y_train[5000])
    #print(clf.predict(X_train[5000].reshape(1, -1)))

    if runXVal:
        #xValOwnImpl(X_train, y_train_label, clf)
        xValAcc = cross_val_score(clf, X_train, y_train_label, cv=3, scoring="accuracy")
        print("X-validation accuracy score: ", xValAcc)
    if runConfusionMatrix:
        y_train_pred = cross_val_predict(clf, X_train, y_train_label, cv=3)
        cm = confusion_matrix(y_train_label, y_train_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
        disp.plot()
        prec = precision_score(y_train_label, y_train_pred)
        recall = recall_score(y_train_label, y_train_pred)
        f1 = f1_score(y_train_label, y_train_pred)
        plt.title("Precision: %.2f, Recall: %.2f, f1: %.2f" % (prec, recall, f1))
        plt.show()


def main():
    dataClass = prepareData()
    X_train = dataClass.X_train
    y_train = dataClass.y_train
    X_test = dataClass.X_test
    y_test = dataClass.y_test

    #print(X_train, y_train)
    #dataClass.plotDigit(data=X_train, label=y_train, number=36000, show=True)

    binaryClass(X_train, y_train)

if __name__ == "__main__":
    t0 = time.time()
    main()
    run_time = time.time() - t0
    print("Run time in %.3f s" % run_time)

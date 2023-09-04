import os
import tarfile
from six.moves import urllib


def fetchData(url: str = None, path: str = None):
    """
    create local directory for the output given path argument
    download data given as tar ball given a url argument
    extract tar ball
    """
    if not os.path.isdir(path):
        os.makedirs(path)
    tgzPath = os.path.join(path, url.split("/")[-1])
    urllib.request.urlretrieve(url, tgzPath)
    tgz = tarfile.open(tgzPath)
    tgz.extractall(path=path)
    tgz.close()
    print("fetching data succesful")


import pandas as pd
import pyspark.pandas as ps
import gzip


def loadDataAsDF(pathToFile: str = None, spark=True):
    """
    load csv file as pandas dataframe given a local path and a filename
    bool spark enables to return a pyspark dataframe instead of a pandas dataframe
    pyspark requires java to work
    """
    print(f"load data as dataframe from {pathToFile}")
    try:
        df = pd.read_csv(pathToFile)
    except:
        with gzip.open(pathToFile, "rt", encoding="utf-8") as f:
            df = pd.read_csv(f)
    if spark:
        sparkDF = ps.from_pandas(df)
        df = sparkDF
    return df


def inspectDFCol(df: pd.DataFrame = None, coloumn: str = ""):
    """
    inspect a dataframe column
    different values and count
    """
    val_count = df[coloumn].value_counts()
    print("\nvalue count: \n", val_count)


import matplotlib.pyplot as plt


def quickHistogram(df: pd.DataFrame = None):
    """
    create a quick and dirty histogram of all dataframe columns
    """
    df.hist(bins=50, figsize=(20, 15))
    plt.show()


import numpy as np


def splitTrainTest(df: pd.DataFrame = None, ratio=0.2):
    """
    manually split a dataframe into a train and test set
    returns train and test df
    ratio is the relative of test to train set
    """
    np.random.seed(42)
    shuffled = np.random.permutation(len(df))
    testSetSize = int(len(df) * ratio)
    testIndices = shuffled[:testSetSize]
    trainIndices = shuffled[testSetSize:]
    retValTrain = df.iloc[trainIndices]
    retValTest = df.iloc[testIndices]
    return retValTrain, retValTest

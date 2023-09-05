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


import numpy as np


def splitTrainTest(df: pd.DataFrame = None, ratio=0.2):
    """
    manually random split of a dataframe into a train and test set
    returns train and test df
    ratio is the relative of test to train set
    problem: generates always a new test set, over time we will see all data when we re-run the script
    """
    np.random.seed(42)
    shuffled = np.random.permutation(len(df))
    testSetSize = int(len(df) * ratio)
    testIndices = shuffled[:testSetSize]
    trainIndices = shuffled[testSetSize:]
    retValTrain = df.iloc[trainIndices]
    retValTest = df.iloc[testIndices]
    return retValTrain, retValTest


import hashlib


def splitTrainTestByID(df, ratio, id_col="index", hash=hashlib.md5):
    """
    advanced random splitting of training and test data set
    preserve same splitting of data even when re-running using some properties of the entries
    """

    def testSetCheck(identifier, ratio, hash):
        """
        get hash of each instance’s identifier, keep only last byte of hash, put instance in test set if this value is e.g. lower or equal to 51 (~20 percent of 256)”
        """
        i = np.int64(identifier)
        h = hash(i).digest()[-1] < (256 * ratio)
        return h

    ids = df[id_col]
    inTestSet = ids.apply(lambda id_: testSetCheck(id_, ratio, hash))
    retValTest = df.loc[inTestSet]
    retValTrain = df.loc[~inTestSet]
    return retValTrain, retValTest


from sklearn.model_selection import train_test_split


def splitTrainTestSKLearn(df, ratio):
    """
    split data in training and test set randomly
    random valid if data set is large enough
    ratio is relative test size
    """
    retValTrain, retValTest = train_test_split(df, test_size=ratio, random_state=42)
    return retValTrain, retValTest


# stratified sampling
# divide opulation into homogeneous subgroups called strata
# ensure test and training sets are representatives of total population
# stratum could be bin of histogram and each bin should have entries in training and test set to be representative


def coloumnToCategory(
    df: pd.DataFrame = None,
    cat: str = "",
    col: str = "",
    divider: float = 0.0,
    merge_val: float = 0.0,
    replace_val: float = 0.0,
):
    """
    get new category attribute by dividing the median income by 1.5 (to limit the number of income categories), and rounding up using ceil (to have discrete categories), and then merging all the categories greater than 5 into category 5

    df.where: entries where cond is False are replaced with corresponding value from other.

    usage:
    coloumnToCategory(
        df, cat="income_cat", col="median_income", divider=1.5, merge_val=5.0, replace_val=5.0
    )
    """

    df[cat] = np.ceil(df[col] / divider)
    if merge_val > 0 and replace_val > 0:
        df[cat].where(cond=df[cat] < merge_val, other=merge_val, inplace=True)
    return df


from sklearn.model_selection import StratifiedShuffleSplit


def stratSplit(df, ratio=0.2, cat: str = None):
    split = StratifiedShuffleSplit(n_splits=1, test_size=ratio, random_state=42)
    for train_index, test_index in split.split(df, df[cat]):
        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]
    return strat_train_set, strat_test_set


def testTrainSplit(df: pd.DataFrame = None, method: str = ""):
    """
    wrapper to get test and train datasets from dataframe with various methods
    """

    if method == "method1":
        df_train, df_test = splitTrainTest(df, 0.2)

    elif method == "method2":

        df_train, df_test = splitTrainTestByID(df, ratio=0.2, id_col="index")

        # testing the train test split
        print("train set: ", len(df_train))
        print("test set: ", len(df_test))
        print("ratio test/train: %.2f " % float(len(df_test) / len(df_train)))

    elif method == "method3":

        df = coloumnToCategory(
            df,
            cat="income_cat",
            col="median_income",
            divider=1.5,
            merge_val=5.0,
            replace_val=5.0,
        )
        df_train, df_test = stratSplit(df=df, ratio=0.2, cat="income_cat")

        print("total set: ", df["income_cat"].value_counts() / len(df))
        print("train set: ", df_train["income_cat"].value_counts() / len(df_train))
        print("test set: ", df_test["income_cat"].value_counts() / len(df_test))

        # agian remove the new category used for splitting
        for set in (df_train, df_test):
            set.drop(["income_cat"], axis=1, inplace=True)

    return df_train, df_test


import matplotlib.pyplot as plt
import seaborn as sns


def visuals(df):
    """
    scatter plot and other visuals
    """
    df = df.copy()
    print(df.head())
    print(df.columns)

    def scatter(xcol, ycol):
        df.plot(
            kind="scatter",
            x=xcol,
            y=ycol,
            alpha=0.5,
            s=df["population"] / 100,  # size of blobs
            label="population",
            c="median_house_value",  # color axis
            cmap=plt.get_cmap("jet"),
            colorbar=True,
        )
        plt.legend()
        plt.show()

    def scatterSeaborn(
        df: pd.DataFrame = None,
        xcol: str = "",
        ycol: str = "",
        size_col: str = "",
        title: str = "",
    ):
        sns.scatterplot(
            data=df,
            x=xcol,
            y=ycol,
            size=size_col,
            hue=size_col,
            palette="viridis",
            alpha=0.5,
        )
        plt.legend(title=size_col, bbox_to_anchor=(1.05, 0.95), loc="upper left")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def quickHistogram(df: pd.DataFrame = None):
        """
        create a quick and dirty histogram of all dataframe columns
        """
        # df.hist(bins=50, figsize=(20, 15))
        df.hist(figsize=(12, 10), bins=30, edgecolor="black")
        plt.subplots_adjust(hspace=0.7, wspace=0.4)
        plt.show()

    quickHistogram(df)
    scatter(xcol="longitude", ycol="latitude")
    scatterSeaborn(
        df=df,
        xcol="longitude",
        ycol="latitude",
        size_col="median_house_value",
        title="Median house value depending of\n their spatial location",
    )


def correlationMatrix(df):
    """
    get correlation matrix of a dataframe
    """
    corr_matrix = df.corr()
    return corr_matrix


from pandas.plotting import scatter_matrix


def scatterMatrix(df, attr):
    """
    shows scatter plot for various numerical values
    """
    scatter_matrix(df[attr], figsize=(12, 8))
    plt.show()


def correlations(df):

    corr = correlationMatrix(df)
    medHouseValCorr = corr["median_house_value"].sort_values(ascending=False)
    print(medHouseValCorr)

    scatterM = scatterMatrix(
        df=df_dftrain,
        attr=[
            "median_house_value",
            "median_income",
            "total_rooms",
            "housing_median_age",
        ],
    )


def addingFeatures(df):
    """
    adding some features to the dataframe based on column information, e.g. ratios of two columns
    """
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]
    return df


def labelPredictorSplit(df: pd.DataFrame = None, target: str = None):
    """
    separate predictors and label in dataframe
    """
    df_pred = df.drop(target, axis=1)
    df_labels = df[target].copy()
    return df_pred, df_labels


def dataCleaningNans(
    df: pd.DataFrame = None, col_with_nan: str = None, method: str = None
):
    """
    clean the data manually
    method1: drop nans (rows)
    method2: drop columns with nans
    method3: set missing values to some value (e.g. median)
    """
    if method == "method1":
        df = df.dropna(subset=[col_with_nan])
    elif method == "method2":
        df = df.drop(col_with_nan, axis=1)
    elif method == "method3":
        # replace nan with e.g. median
        median = df[col_with_nan].median()
        df[col_with_nan].fillna(median)
        # if this is training set, we would need the median later to replace missing values in test set as well
    return df


from sklearn.impute import SimpleImputer
import numpy as np


def dataCleaningNansImputer(df):
    """
    use scikit-learns imputer to replace missing values e.g. using the median
    make sure the dataframe used here only contains numerical values, i.e. drop non-numericals first, see try-except-block
    the imputer is kind of trained with the training and can be applied as an estimator also to the test set:
    X = imputer.transform(df_test_num)
    df_tr = pd.DataFrame(X, columns=df_num.columns)
    """
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    # imputer can only handle numerical data!
    try:
        df_num = df.drop("ocean_proximity", axis=1)
    except:
        df_num = df
    imputer.fit(df_num)
    # imputer.statistics_
    # print(df_num.median().values)

    df = pd.DataFrame(imputer.transform(df_num), columns=df_num.columns)
    return df


def cleanNans(df: pd.DataFrame = None, method: str = None):
    """
    wrapper for cleaning nans in dataframe
    """
    if method == "method1":
        df = dataCleaningNans(df, col_with_nan="total_bedrooms", method="method1")
    elif method == "method2":
        df = dataCleaningNansImputer(df)
    return df

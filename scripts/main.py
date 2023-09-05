from dataOrganizer import (
    fetchData,
    loadDataAsDF,
    inspectDFCol,
    quickHistogram,
    splitTrainTest,
    splitTrainTestByID,
    coloumnToCategory,
    stratSplit,
)

import pandas as pd

def readingData(verbose=True, plotting=True):
    # define path for download
    ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    PATH = "datasets/housing"
    FILE = "/housing.tgz"
    URL = ROOT + PATH + FILE
    fetchData(url=URL, path=PATH)
    df = loadDataAsDF(pathToFile=PATH + FILE, spark=False)
    if verbose:
        print(df.head(10))
        df.info()
        print("Type: %s " % type(df))

        inspectDFCol(df, "ocean_proximity")

        print(df.describe())

    if plotting:
        quickHistogram(df)

    # add index
    df = df.reset_index()
    return df


def testTrainSplit(df: pd.DataFrame=None, method: str=""):
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
            df, cat="income_cat", col="median_income", divider=1.5, merge_val=5.0, replace_val=5.0
        )
        df_train, df_test = stratSplit(df=df, ratio=0.2, cat="income_cat")

        print("total set: ", df["income_cat"].value_counts() / len(df))
        print("train set: ", df_train["income_cat"].value_counts() / len(df_train))
        print("test set: ", df_test["income_cat"].value_counts() / len(df_test))

        # agian remove the new category used for splitting 
        for set in (df_train, df_test):
            set.drop(["income_cat"], axis=1, inplace=True)

    return df_train, df_test


def main():
    df = readingData(verbose=False, plotting=False)
    print(df.head())
    df_test, df_train = testTrainSplit(df, method="method3")




if __name__ == "__main__":
    main()

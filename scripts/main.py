from dataOrganizer import (
    fetchData,
    loadDataAsDF,
    inspectDFCol,
    quickHistogram,
    splitTrainTest,
)


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

    return df


def main():
    df = readingData(verbose=False, plotting=False)
    df_train, df_test = splitTrainTest(df, 0.2)
    # testing the train test split
    print("train set: ", len(df_train))
    print("test set: ", len(df_test))
    print("ratio test/train: %.2f " % float(len(df_test) / len(df_train)))


if __name__ == "__main__":
    main()

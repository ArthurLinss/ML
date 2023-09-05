from dataOrganizer import (
    fetchData,
    loadDataAsDF,
    inspectDFCol,
    testTrainSplit,
    visuals,
    correlations,
    addingFeatures,
    labelPredictorSplit,
    cleanNans,
)


def readingDataFromWeb(verbose=True, plotting=True):
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

    # add index
    df = df.reset_index()
    return df


def readingDataFromSciKit():
    from sklearn.datasets import fetch_california_housing

    # https://inria.github.io/scikit-learn-mooc/python_scripts/datasets_california_housing.html

    df = fetch_california_housing(as_frame=True)
    df = df.frame
    d = {
        "MedInc": "median_income",
        "HouseAge": "housing_median_age",
        "AveRooms": "total_rooms",
        "AveBedrms": "total_bedrooms",
        "Population": "population",
        "AveOccup": "households",
        "Latitude": "latitude",
        "Longitude": "longitude",
        "MedHouseVal": "median_house_value",
    }
    df.rename(columns=d, inplace=True)
    return df


import pandas as pd


def main():
    # df = readingDataFromWeb(verbose=False, plotting=False)
    df = readingDataFromSciKit()
    df = addingFeatures(df)
    print(df.columns)
    print(df.head())
    df_train, df_test = testTrainSplit(df, method="method3")

    visuals(df_train)
    correlations(df_train)
    df_train = cleanNans(df_train)

    pred, label = labelPredictorSplit(df=df_train, target="median_house_value")


if __name__ == "__main__":
    main()

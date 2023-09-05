from dataOrganizer import (
    fetchData,
    loadDataAsDF,
    inspectDFCol,
    quickHistogram,
    splitTrainTest,
    splitTrainTestByID,
    coloumnToCategory,
    stratSplit,
    testTrainSplit,
    visuals,
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


def readingDataFromSciKit():
    from sklearn.datasets import fetch_california_housing

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


def main():
    # df = readingData(verbose=False, plotting=False)
    df = readingDataFromSciKit()
    print(df.columns)
    print(df.head())
    df_train, df_test = testTrainSplit(df, method="method3")

    # do not touch test set
    visuals(df_train)


if __name__ == "__main__":
    main()

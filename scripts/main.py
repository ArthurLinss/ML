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
    encoder,
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

    # add pseudo categorical data 
    df["ocean_proximity"] = "INLAND"
    for index, row in df.iterrows():
        if index % 5 == 0:
            df.at[index, "ocean_proximity"] = "ISLAND"
        if index % 6 == 0:
            df.at[index, "ocean_proximity"] = "NEAR BAY"
        if index>500:
            break

    return df


def main():
    # df = readingDataFromWeb(verbose=False, plotting=False)
    df = readingDataFromSciKit()
    df = addingFeatures(df)
    df = encoder(df, cat_data="ocean_proximity", method="method3")
    df = cleanNans(df)
    print(df.columns)
    print(df.head())

    df_train, df_test = testTrainSplit(df, method="method3")

    #visuals(df_train)
    #correlations(df_train)

    
    pred, label = labelPredictorSplit(df=df_train, target="median_house_value")


if __name__ == "__main__":
    main()

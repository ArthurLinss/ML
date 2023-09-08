"""
some basic machine learning workflow including
- loading data from web or scikit learn
- prepare data
    - clean nans (different methods)
    - scale training data
    - encode categorical data (one-hot-encoding)
    - split train and test data
    - split label (target) and data
- machine learning models
    - linear regression
    - decision tree
    - random forest
"""

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
    scaling,
)

from models import (
    linearRegression,
    getRMSE,
    decTree,
    crossVal,
    forestReg,
    gridSearch,
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
        if index > 500:
            break

    return df


def main():
    # df = readingDataFromWeb(verbose=False, plotting=False)
    df = readingDataFromSciKit()
    df = addingFeatures(df)

    print(df.columns)
    print(df.head())

    df_train, df_test = testTrainSplit(df, method="method3")

    # visuals(df_train)
    # correlations(df_train)

    df_train, coder = encoder(df_train, cat_data="ocean_proximity", method="method3")
    df_train, scaler = scaling(df=df_train, method="method2")
    df_train = cleanNans(df_train, method="method2")

    print(df_train.head())

    pred, label = labelPredictorSplit(df=df_train, target="median_house_value")

    # linear regression model
    lin_reg = linearRegression(pred, label)
    # test
    rmse = getRMSE(lin_reg, pred.iloc[:5], label.iloc[:5])

    # decision tree model
    tree = decTree(pred, label)
    # test
    rmse = getRMSE(tree, pred.iloc[:5], label.iloc[:5])
    # cross validation of decision tree
    cv = crossVal(tree, pred, label, cv=10, verbose=True)

    # random forest model
    forest = forestReg(pred, label)
    rmse = getRMSE(forest, pred.iloc[:5], label.iloc[:5])
    cv = crossVal(forest, pred, label, cv=10, verbose=True)

    param_grid = [
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]
    gridSearch = gridSearch(
        model=forest,
        data=pred,
        label=label,
        param_grid=param_grid,
    )


if __name__ == "__main__":
    main()

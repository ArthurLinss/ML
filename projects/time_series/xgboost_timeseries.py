"""
XGBoost Regression with Time Series

Resources:
- https://xgboost.readthedocs.io/en/stable/python/python_api.html
- https://machinelearningmastery.com/xgboost-loss-functions/
- https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/
- https://towardsdatascience.com/the-complete-guide-to-time-series-analysis-and-forecasting-70d476bfe775
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import seaborn as sns
import xgboost as xgb
from dateutil.relativedelta import relativedelta
import json
from sklearn.model_selection import TimeSeriesSplit

plt.style.use("fivethirtyeight")
color_pal = sns.color_palette()
print("XGBoost Version: ", xgb.__version__)

"""
config
"""
train_test = True
predict = True

use_legs = False  # check if implementation is correct
exclude_corona = True
use_double_exp_smoothing = False  # needs to be tuned


y_title = "Stückzahl"
x_title = "Zeit"
size_tuple = (17, 5)
"""
loading and preparing data
"""
df = pd.read_csv("groupby_month.csv")
date_column = "Buchungsdatum (BL)_datetime"
df = df.rename(columns={date_column: "date"})
df["date2"] = df["date"]
df = df.set_index("date")
df.index = pd.to_datetime(df.index)


if use_double_exp_smoothing:

    def double_exponential_smoothing(series, alpha, beta):
        result = [series[0]]
        for n in range(1, len(series) + 1):
            if n == 1:
                level, trend = series[0], series[1] - series[0]
            if n >= len(series):  # forecasting
                value = result[-1]
            else:
                value = series[n]
            last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
            trend = beta * (level - last_level) + (1 - beta) * trend
            result.append(level + trend)
        return result

    df["STsmooth"] = double_exponential_smoothing(df["ST"], alpha=0.3, beta=0.5)[:-1]
    df["ST"] = df["STsmooth"]


def create_features(df):
    """
    adding some datetime features to dataframe
    """
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["quarter"] = df.index.quarter
    # df["day"] = df.index.day
    # df["dayofyear"] = df.index.dayofyear
    return df


df = create_features(df)


def create_leg_features(df):
    """
    create lag features in dataframe
    """
    target_map = df["ST"].to_dict()
    df["leg1"] = (df.index - pd.Timedelta(days=365)).map(target_map)
    df["leg2"] = (df.index - pd.Timedelta(days=730)).map(target_map)
    df["leg3"] = (df.index - pd.Timedelta(days=1095)).map(target_map)
    return df


if use_legs:
    df = create_leg_features(df)

print(df.head())
print(df.tail())
print(df.index)


def plotTimeSeriesSimple(df):
    """
    create simple plot of a time series with time as index
    """
    # plt.figure(figsize=(18, 8))
    fig, ax = plt.subplots(figsize=(18, 8))

    if use_double_exp_smoothing:
        df.plot(ax=ax, y="ST", figsize=(15, 5), color=color_pal[0])
        df.plot(ax=ax, y="STsmooth", figsize=(15, 5), color=color_pal[1])
    else:
        df.plot(y="ST", figsize=(15, 5), color=color_pal[0])
        # ax.legend([""])
        # ax.get_legend().remove()

    # plt.xticks(rotation=90)
    plt.title("Zeitreihe")
    plt.ylabel(y_title)
    plt.xlabel(x_title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot" + "_" + "xgboost" + "_" + "TimeSeriesSimple" + ".png")
    plt.clf()


plotTimeSeriesSimple(df)


if train_test:
    print("running train test split")

    split_date = "2020-06-01"
    train = df.loc[df.index <= split_date]
    test = df.loc[df.index >= split_date]

    if exclude_corona:
        split_date1 = "2020-03-01"
        split_date2 = "2020-04-01"
        train = df.loc[df.index <= split_date1]
        test = df.loc[df.index >= split_date2]

    def plotTestTrainSplit(train, test, split_date1=None, split_date2=None):
        """
        plots test and training data in different colors
        training data is used until split_date1
        test data is used starting at split_date2
        """
        train_skimmed = train.drop(columns=["month", "quarter", "year", "date2"])
        test_skimmed = test.drop(columns=["month", "quarter", "year", "date2"])

        print(train_skimmed)
        print(test_skimmed)

        fig, ax = plt.subplots(figsize=size_tuple)
        train_skimmed.plot(ax=ax, label="Training")
        test_skimmed.plot(ax=ax, label="Test")
        # train.plot(label="training")
        # test.plot(label="test")
        if split_date1 != None:
            ax.axvline(split_date1, color="black", ls="--", linewidth=2)
        if split_date2 != None:
            ax.axvline(split_date2, color="black", ls="--", linewidth=2)
        ax.legend(["Trainingsdaten", "Testdaten"])
        plt.title("")
        plt.ylabel(y_title)
        plt.xlabel(x_title)
        plt.tight_layout()
        plt.savefig("plot" + "_" + "xgboost" + "_" + "TrainTestSplit" + ".png")
        plt.clf()

    if exclude_corona:
        plotTestTrainSplit(train, test, split_date1, split_date2)
    else:
        plotTestTrainSplit(train, test, split_date)

    def boxplot(df, x, y="ST"):
        """
        wrapper to create boxplot of a column w.r.t. to time
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.boxplot(data=df, x=x, y=y)
        title = {"month": "Monat", "year": "Jahr", "quarter": "Quartal"}
        final_title = title[x]
        plt.ylabel(y_title)
        plt.xlabel(final_title)
        ax.set_title(f"Boxplot %s" % final_title)
        plt.tight_layout()
        plt.savefig("plot" + "_" + "xgboost" + "_" + "boxplot" + "_" + x + ".png")

    boxplot(df, "month")
    boxplot(df, "year")
    boxplot(df, "quarter")

    # model
    FEATURES = ["month", "quarter", "year"]
    if use_legs:
        FEATURES.extend(["leg1", "leg2", "leg3"])
    TARGET = "ST"
    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_test = test[FEATURES]
    y_test = test[TARGET]

    # regressor
    """
    n_estimators (int) – Number of gradient boosted trees. Equivalent to number of boosting rounds.
    max_depth (Optional[int]) – Maximum tree depth for base learners.
    max_leaves – Maximum number of leaves; 0 indicates no limit.
    max_bin – If using histogram-based algorithm, maximum number of bins per feature
    grow_policy – Tree growing policy. 0: favor splitting at nodes closest to the node, i.e. grow depth-wise. 1: favor splitting at nodes with highest loss change.
    learning_rate (Optional[float]) – Boosting learning rate (xgb’s “eta”)
    verbosity (Optional[int]) – The degree of verbosity. Valid values are 0
    """
    reg = xgb.XGBRegressor(
        base_score=0.5,
        booster="gbtree",
        n_estimators=1000,
        early_stopping_rounds=50,
        objective="reg:linear",
        # objective="reg:squarederror",
        max_depth=3,
        learning_rate=0.01,
    )  # try avoid overfitting
    reg.fit(
        X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=100
    )
    reg.save_model("model_train_test.json")

    # feature importance
    def plotFeatureImportance(reg):
        """
        plot feature importance as horizontal bar plot
        """
        df_reg = pd.DataFrame(
            data=reg.feature_importances_,
            index=reg.feature_names_in_,
            columns=["importance"],
        )
        df_reg = df_reg.sort_values("importance")
        fig, ax = plt.subplots(figsize=size_tuple)
        df_reg.plot(kind="barh", title="Feature Ranking")
        plt.tight_layout()
        plt.grid(True)
        plt.savefig("plot" + "_" + "xgboost" + "_" + "FeatureImportance" + ".png")
        plt.clf()

    plotFeatureImportance(reg)

    # forecast on test, create future dates for each month
    test["prediction"] = reg.predict(X_test)
    df = df.merge(test[["prediction"]], how="left", left_index=True, right_index=True)
    print("future dataframe \n", df)

    # score
    score = np.sqrt(mean_squared_error(test["ST"], test["prediction"]))
    print(f"RMSE Score on Test set: {score:0.2f}")
    # error (Look at the worst and best predicted days)
    test["error"] = np.abs(test[TARGET] - test["prediction"])
    test["date3"] = test.index.date
    print(test.groupby(["date3"])["error"].mean().sort_values(ascending=False).head(10))

    def plotDataVsPrediction(df, kwargs={}):
        fig, ax = plt.subplots(figsize=size_tuple)
        df[["ST"]].plot(ax=ax)
        df["prediction"].plot(ax=ax)
        plt.legend(["Daten", "Vorhersage"])
        ax.set_title("Daten und Vorhersage")
        plt.ylabel(y_title)
        plt.xlabel(x_title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("plot" + "_" + "xgboost" "_" + "DataVsPrediction" + ".png")
        plt.clf()

    plotDataVsPrediction(df, kwargs={"score": score})

    def plotDataVsPredictionPeriod(df, start_date="2020-06-01", end_date="2021-06-01"):
        """
        plot truth data vs prediction for some specific period
        """
        fig, ax = plt.subplots(figsize=size_tuple)
        df.loc[(df.index > start_date) & (df.index < end_date)]["ST"].plot(
            ax=ax, figsize=(15, 5), title="Week Of Data"
        )
        df.loc[(df.index > start_date) & (df.index < end_date)]["prediction"].plot(
            ax=ax, style="."
        )
        plt.legend(["Truth Data", "Prediction"])
        plt.savefig(
            "plot" + "_" + "xgboost" + "_" + "DataVsPredictionPeriod" + "_" + ".png"
        )
        plt.clf()

    """
    next steps:
    - better cross-validation
    - parameter tuning
    - more features (number of public holidays per month, holidas )
    - exclude certain time ranges with unnormal behaviour
    """


if predict == True:
    print("running prediction")
    n_month_future = 6

    start = df.index.max()
    end = df.index.max() + relativedelta(months=n_month_future)
    future = pd.date_range(start, end, freq="MS", inclusive="right")
    future_df = pd.DataFrame(index=future)
    future_df["date2"] = future_df.index
    future_df["date2"] = future_df["date2"].apply(lambda x: x.strftime("%Y-%m"))
    future_df["is_future"] = True
    df["is_future"] = False

    df_and_future = pd.concat([df, future_df])
    df_and_future = create_features(df_and_future)
    print(df_and_future)
    print(df_and_future.tail(8))

    future_with_features = df_and_future.query("is_future").copy()

    FEATURES = ["month", "quarter", "year"]
    if use_legs:
        FEATURES.extend(["leg1", "leg2", "leg3"])
    TARGET = "ST"
    X_all = df[FEATURES]
    y_all = df[TARGET]

    # regressor
    # parameter tuning possible (holidays etc)
    reg = xgb.XGBRegressor(
        base_score=0.5,
        booster="gbtree",
        n_estimators=1000,
        early_stopping_rounds=50,
        objective="reg:linear",
        max_depth=3,
        learning_rate=0.01,
    )  # try avoid overfitting
    reg.fit(X_all, y_all, eval_set=[(X_all, y_all), (X_all, y_all)], verbose=100)
    reg.save_model("model.json")
    # reg_new = xgb.XGBRegressor()
    # reg_new.load_model("model.json")
    reg_config = xgb.get_config()
    with open("reg_config.txt", "w") as file:
        file.write(json.dumps(reg_config))

    future_with_features["pred"] = reg.predict(future_with_features[FEATURES])

    def plotFuturePrediction(df):
        fig, ax = plt.subplots(figsize=size_tuple)
        df.plot(ax=ax, y="pred")
        plt.title("Vorhersage")
        plt.ylabel(y_title)
        plt.xlabel(x_title)
        ax.legend([""])
        ax.get_legend().remove()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("plot" + "_" + "xgboost" + "_" + "FuturePrediction" + ".png")
        plt.clf()

    plotFuturePrediction(future_with_features)

    def plotFuturePredictionAppend(df1, df2):
        print(df1)
        print(df2)
        df1 = pd.concat([df1, df2])

        fig, ax = plt.subplots(figsize=size_tuple)
        df1.plot(ax=ax, y="pred")
        df2.plot(ax=ax, y="pred")
        plt.title("Vorhersage")
        plt.ylabel(y_title)
        plt.xlabel(x_title)
        ax.legend([""])
        ax.get_legend().remove()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("plot" + "_" + "xgboost" + "_" + "FuturePredictionAppend" + ".png")
        plt.clf()

    df["pred"] = df["ST"]
    plotFuturePredictionAppend(df, future_with_features)

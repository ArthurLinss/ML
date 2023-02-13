#  help: https://datagy.io/python-sklearn-linear-regression/

import pandas as pd
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", "--p", action="store_true", help="show plots")
    args = parser.parse_args()
    return args

def linreg():
    """
    linear regression only non-smokers and one input variable
    """
    print("========== first run ===========")

    args = parser()

    df = pd.read_csv('https://raw.githubusercontent.com/datagy/data/main/insurance.csv')
    print(df.head())
    #print(df.info())
    print("correlation all:", df.corr())

    # plotting
    split_by = 'smoker'
    p = lambda split : sns.pairplot(df, hue=split)
    p(split_by)
    if args.p:
        plt.show()


    sns.relplot(data=df, x='age', y='charges', hue=split_by)
    if args.p:
        plt.show()

    non_smokers = df[df['smoker'] == 'no']
    print("correlation non-smokers: ", non_smokers.corr())



    model = LinearRegression()

    # X -> features, array of shape n samples, m features
    # y -> array of shape n samples

    X = non_smokers[['age']]
    y = non_smokers[['charges']]

    print("types X,y: ", type(X), type(y))

    # split into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.3)

    # fit the model to train data only
    model.fit(X_train, y_train)

    # use test data in pred
    pred = model.predict(X_test)

    # rsquared test -> proportion of variance explained by features
    r2 = r2_score(y_test, pred)
    rmse = mean_squared_error(y_test, pred, squared=False)

    print("r2: ", r2)
    print("rmse: ", rmse)


def linreg2():
    print("========== 2nd run ===========")


    args = parser()
    df = pd.read_csv('https://raw.githubusercontent.com/datagy/data/main/insurance.csv')
    non_smokers = df[df['smoker'] == 'no']
    X = non_smokers[["age","bmi"]]
    y = non_smokers[["charges"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.3)

    model2 = LinearRegression()
    model2.fit(X_train, y_train)

    pred = model2.predict(X_test)
    r2 = r2_score(y_test, pred)
    rmse = mean_squared_error(y_test, pred, squared=False)

    print("the r2 is: ", r2)
    print("the rmse is: ", rmse)


def linreg3():
    print("========== 3d run ===========")


    args = parser()
    df = pd.read_csv('https://raw.githubusercontent.com/datagy/data/main/insurance.csv')

    # make smoker numerical
    df["smoker_int"] = df["smoker"].map({"yes":1, "no":0})

    print(df.head())

    X = df[["age","bmi","smoker_int"]]
    y = df[["charges"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.3)

    model3 = LinearRegression()
    model3.fit(X_train, y_train)

    pred = model3.predict(X_test)
    r2 = r2_score(y_test, pred)
    rmse = mean_squared_error(y_test, pred)

    print("the r2 is: ", r2)
    print("the rmse is: ", rmse)

    print("coef: ", model3.coef_)
    print("intercet: ", model3.intercept_)


    # Writing a function to predict charges
    coefficients = model3.coef_[0]
    intercept = model3.intercept_
    def calculate_charges(age, bmi, smoker):
        return (age * coefficients[0]) + (bmi * coefficients[1]) + (smoker * coefficients[2]) + intercept

    # test
    print("test calc.: ", calculate_charges(33, 22, 0))



def linreg4():
    print("========== 4th run ===========")


    args = parser()
    df = pd.read_csv('https://raw.githubusercontent.com/datagy/data/main/insurance.csv')
    df.dropna()

    # make smoker numerical
    df["smoker_int"] = df["smoker"].map({"yes":1, "no":0})
    df["sex_int"] = df["sex"].map({"male":1, "female":0})

    print(df.head())

    X = df[["age","bmi","smoker_int","sex_int"]]
    y = df[["charges"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.3)

    model3 = LinearRegression()
    model3.fit(X_train, y_train)

    pred = model3.predict(X_test)
    r2 = r2_score(y_test, pred)
    rmse = mean_squared_error(y_test, pred)

    print("the r2 is: ", r2)
    print("the rmse is: ", rmse)

    print("coef: ", model3.coef_)
    print("intercet: ", model3.intercept_)


    # Writing a function to predict charges
    coefficients = model3.coef_[0]
    intercept = model3.intercept_
    def calculate_charges(age, bmi, smoker):
        return (age * coefficients[0]) + (bmi * coefficients[1]) + (smoker * coefficients[2]) + intercept

    # test
    print("test calc.: ", calculate_charges(33, 22, 0))

if __name__ == "__main__":
    linreg()
    linreg2()
    linreg3()
    linreg4()

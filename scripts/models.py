import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression


def linearRegression(data, label):
    """
    linear regression model
    lin_reg.predict(some_data)
    """
    lin_reg = LinearRegression()
    lin_reg.fit(data, label)
    print("Linear Regression Coeff.: ", lin_reg.coef_)
    return lin_reg


from sklearn.metrics import mean_squared_error


def getRMSE(model, data, labels):
    """
   root mean squared error 
   """
    pred = model.predict(data)
    mse = mean_squared_error(labels, pred)
    rmse = np.sqrt(mse)
    print("predictions: ", pred)
    print("labels: ", labels)
    print("rmse: ", rmse)
    return rmse


# RMSE(lin_reg, pred, labels)


from sklearn.tree import DecisionTreeRegressor


def decTree(data, label):
    """
   decision tree regressor
   tends to overfitting
   use cross validation!
   """
    tree = DecisionTreeRegressor()
    tree.fit(data, label)
    return tree


from sklearn.model_selection import cross_val_score


def crossVal(model, data, label, cv=10, verbose=True):
    """
   cross validation:
   split training set into smaller piceces (folds)
   trains and evaluates model x (10) times picking different folds for evaluation
   returns scores for each evaluation
   solutinos for overfitting: simplify the model, constrain/regularize it, get more training data
   """
    scores = cross_val_score(
        model, data, label, scoring="neg_mean_squared_error", cv=cv
    )
    rmse_scores = np.sqrt(-scores)
    if verbose:
        print("Scores:", rmse_scores)
        print("Mean:", rmse_scores.mean())
        print("Standard deviation:", rmse_scores.std())
    return rmse_scores


from sklearn.ensemble import RandomForestRegressor


def forestReg(data, label):
    """
   random forest (multiple decision trees) as ensemble learner
   """
    forest = RandomForestRegressor()
    forest.fit(data, label)
    return forest


import joblib


def saveModel(model, name):
    """
   save a trained model
   """
    joblib.dump(model, name + ".pkl")
    print("saved model")


def loadModel(model, name):
    """
   load a trained model
   """
    print("loading model ...")
    model = joblib.load(name + ".pkl")
    return model


from sklearn.model_selection import GridSearchCV


def gridSearch(
    model,
    data,
    label,
    param_grid=param_grid,
):
    """
   optimize model's hyperparameters via grid search by testing different combinations of hyperparemeters
   hyperparameters: if no knowledge, try powers of 10
   """
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring="neg_mean_squared_error"
    )
    grid_search.fit(data, label)
    print("Grid search best param: ", grid_search.best_params_)
    return grid_search

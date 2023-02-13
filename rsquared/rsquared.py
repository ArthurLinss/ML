"""
linear regression and rsquared (coefficient of determination)
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
- resource: https://www.statology.org/r-squared-in-python/

- rsquared = proportion of the variance in the response variable that can be explained by the predictor variable


"""

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def LinReg():
    # number of hours learned
    hours = [1, 2, 2, 4, 2, 1, 5, 4, 2, 4, 4, 3, 6]
    # number of taken preparation exams
    prep = [1, 3, 3, 5, 2, 2, 1, 1, 0, 3, 4, 3, 2]
    # result
    score = [76, 78, 85, 88, 72, 69, 94, 94, 88, 92, 90, 75, 96]

    df = pd.DataFrame({"hours":hours, "prep":prep, "score":score})



    model = LinearRegression()

    # input/predictor and target/response variable
    X = np.array(df[["hours"]], dtype=np.float64)
    y = np.array(df.score, dtype=np.float64)
    print("X shape: ", X.shape, "y shape: ", y.shape)

    model.fit(X,y)
    rsquared = model.score(X, y)
    coef = model.coef_
    print(f"rsquared: {rsquared}, coef: {coef}")
    prediction = model.predict(X)
    print("prediction: ", prediction.shape)


    plt.scatter(X, y)
    plt.plot(X,prediction)
    plt.show()

if __name__ == "__main__":
    LinReg()

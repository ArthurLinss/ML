import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.metrics import mean_squared_error
#import seaborn as sns
#import xgboost as xgb
from dateutil.relativedelta import relativedelta
import json
#from sklearn.model_selection import TimeSeriesSplit

plt.style.use("fivethirtyeight")
#color_pal = sns.color_palette()
#print("XGBoost Version: ", xgb.__version__)
from sklearn.metrics import mean_squared_error



from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance


df = pd.read_csv("groupby_day_full.csv")
split_date = "2020-07"
train = df.loc[df.ym <= split_date]
test = df.loc[df.ym >= split_date]

print("mean: ", df["delta_t"].mean())


FEATURES = [ "m","Umsatzgebiet","Baureihe VT","Verkauf aus Bestand","Kundengruppe"]
CFEATURES = ["Umsatzgebiet","Baureihe VT","Verkauf aus Bestand","Kundengruppe"]



TARGET = "delta_t"
X_train = train[FEATURES]
y_train = train[TARGET]
X_test = test[FEATURES]
y_test = test[TARGET]


print("\n \n \n \n ")
for x in CFEATURES:
    X_train[x] = pd.Categorical(X_train[x])
    X_test[x] = pd.Categorical(X_test[x])
print(X_train.dtypes)


"""
#from sklearn.compose import ColumnTransformer
#from sklearn.preprocessing import OneHotEncoder
 # one-hot encode the categorical features
#cat_attribs = FEATURES
#full_pipeline = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat_attribs)], remainder='passthrough')
#encoder = full_pipeline.fit(X_train)
#X_train = encoder.transform(X_train)
#X_test = encoder.transform(X_test)



reg = xgb.XGBRegressor(
    enable_categorical=True,
    base_score=0.5,
    booster="gbtree",
    n_estimators=1000,
    early_stopping_rounds=50,
    objective="reg:linear",
    # objective="reg:squarederror",
    max_depth=3,
    learning_rate=0.01,
)

reg.fit(
    X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=100
)
reg.save_model("model_train_test.json")
"""



from catboost import CatBoostClassifier, CatBoostRegressor
import catboost


model = CatBoostRegressor(iterations=150,
                          learning_rate=0.02,
                          depth=6,
                          cat_features=FEATURES,
                          loss_function="RMSE")
# Fit model
model.fit(X_train, y_train)

# Get predictions
pred = model.predict(X_test)


score = np.sqrt(mean_squared_error(y_test, pred))
print(score)

feature_importane = model.get_feature_importance(prettified=True)

print(feature_importane)

pred = model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(y_test, pred)))
r2 = r2_score(y_test, pred)
print("Testing performance")
print("RMSE: {:.2f}".format(rmse))
print("R2: {:.2f}".format(r2))


import matplotlib.pyplot as plt

plt.scatter(pred, y_test) # plotting t, a separately
plt.show()

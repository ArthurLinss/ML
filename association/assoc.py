import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import LabelEncoder


def association():
    # load data
    url = "https://raw.githubusercontent.com/Prof-Rodrigo-Silva/ScriptR/master/R%20-%20Avan%C3%A7ado%20-%20Data%20Mining/GroceryStoreDataSet.csv"
    df = pd.read_csv(url)

    # encode strings to numerical values
    df = df.apply(LabelEncoder().fit_transform)

    # apriori model
    print(" --- apriori ---")
    df_apr = apriori(df, min_support = 0.2, use_colnames = True, verbose = 1)
    print(df_apr)

    #assoc rule for interpretation
    print(" --- association rules ---")
    df_ar = association_rules(df_apr, metric = "confidence", min_threshold = 0.6)
    print(df_ar)



def main():
    association()

if __name__ == "__main__":
    main()

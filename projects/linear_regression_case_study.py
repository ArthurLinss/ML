#!/usr/bin/env python3

"""
- script to evaluate the candy data set (https://raw.githubusercontent.com/fivethirtyeight/data/master/candy-power-ranking/candy-data.csv)
- just run this python script via: $ python case_study.py
- the results are saved in a newly created subdirectory
"""

import os
import pickle
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from scipy.optimize import linprog


def main():

    # ------------------------------------
    # preparation
    # ------------------------------------
    # create directory for output
    output_dir = "output/"
    create_outputdir(path=output_dir)


    # ------------------------------------
    # load and prepare raw data
    # ------------------------------------
    url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/candy-power-ranking/candy-data.csv"
    df = pd.read_csv(url)
    df.winpercent = df.winpercent/100

    # add new coloumn with winpercent above a certain threshold as bool (finally not needed/used, idea was to use kind of a classifier using that variable)
    # i.e. if winpercent>=threshold, then winpercentbool==1, else 0
    threshold = 0.6
    conditions = [df.winpercent>=threshold, df.winpercent<threshold]
    choices = [1,0]
    df["winpercentbool"] = np.select(conditions, choices)

    # save data set locally to excel
    #df.to_excel(output_dir + "data_local.xlsx")
    # ------------------------------------


    # ------------------------------------
    # first data checks and plots for visualisazion
    # ------------------------------------
    #df.describe()
    #print(df.head(10))
    n_candies = len(df.competitorname.unique())
    print("Number of candies: ", n_candies)
    features = [x for x in df.columns]
    print("Features: ", features)

    # plot distributions
    stacked_histograms(df=df, savename=output_dir+"stacked_vs_")
    # correlation plot
    corr_plot(df=df, savename=output_dir + "heatmap")
    # boxplot to show correlation
    create_boxplots(df=df, savename=output_dir + "boxplot")

    # print only correlations of features with winpercent
    print(df.corr()["winpercent"].sort_values(ascending=False))
    # sort candies by win and create bar plot
    df_sorted_win = df.sort_values(by=['winpercent'], ascending=False)
    bar_plot(df=df_sorted_win, savename=output_dir+"barplot")

    # special scatter plots
    scatter_plot(df=df, savename=output_dir+"scatterplot")
    scatter_plot_win_price(df=df, savename=output_dir+"scatterplot_win_price")
    scatter_plot_win_sugar(df=df, savename=output_dir+"scatterplot_win_sugar")
    # ------------------------------------


    # ------------------------------------
    # main model for linear regression
    # ------------------------------------
    lin_reg(df=df, savename=output_dir)
    # ------------------------------------


    # ------------------------------------
    # Residual calculations and write result to output
    # ------------------------------------
    pricepercent_median = df['pricepercent'][(df["chocolate"]==1) & (df["peanutyalmondy"]==1) ].median()
    sugarpercent_median = df['sugarpercent'][(df["chocolate"]==1) & (df["peanutyalmondy"]==1) ].median()
    print("Median Preis: ", pricepercent_median)
    print("Median Zucker: ", sugarpercent_median)


    with open(output_dir + 'results.txt', 'w') as f:
        f.write(f"\nNumber of candies: {n_candies}")
        f.write(f"\nFeatures: {features}")
        f.write(f"\nMedian Preis:  {pricepercent_median}")
        f.write(f"\nMedian Zucker: {sugarpercent_median}")

    print("\n\n--- Success ---\n\n")
    # ------------------------------------

# ------------------------------------
# below follow all the helper functions
# for analysis, output and plotting
# ------------------------------------

def lin_reg(df: pd.DataFrame, savename="results"):
    """
    create mathematical model, set features as well as target and perform multiple linear regression
    """

    # set features
    X = df.drop(columns=["winpercent","competitorname",'winpercentbool'])
    X = X.drop(columns=["pricepercent"])

    # set target
    y = df.winpercent

    # define and train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    # root mean squared error evaluated with test data (model performance)
    predictions = lin_reg.predict(X_test)
    rmse = np.sqrt(sklearn.metrics.mean_squared_error(predictions, y_test))
    print(f"RMS Error: {rmse}")
    print("Linear regression coefficients: ", list(zip(X.columns,list(lin_reg.coef_))), "Intercept: ", lin_reg.intercept_)
    print("N features in model: ", lin_reg.n_features_in_)
    print("Features in model: ", lin_reg.feature_names_in_)

    # get optimal coefficients
    # first using a library
    run_optimisation(lin_reg=lin_reg)
    # second doing it myself
    dic_coeff = dict(zip(X.columns,list(lin_reg.coef_)))
    rounded_coeff = get_optimal_values(df, dic_coeff)
    print("Rounded coefficients: ", rounded_coeff)

    # calc. best winpercent
    rounded_coeff_arr = np.array(rounded_coeff)
    max_winpercent = np.dot(lin_reg.coef_, rounded_coeff_arr) + lin_reg.intercept_
    print("\n##########\nLinear Regression Result\n##########\n")
    print("Maximal value winpercent: ", max_winpercent)
    print("Reached with features/coefficients/best values: ", list(zip(X.columns,lin_reg.coef_,rounded_coeff)))
    print("\n##########\n##########\n")
    write_resulting_parameters(data=list(zip(X.columns,lin_reg.coef_,rounded_coeff)), kwargs=[("MaxValue",max_winpercent),("Intercept",lin_reg.intercept_),("RMSE",rmse),("N features in model: ", lin_reg.n_features_in_)], savename=savename+"results.csv")

    # save model for later
    filename = savename + 'model.sav'
    pickle.dump(lin_reg, open(filename, 'wb'))

def run_optimisation(lin_reg):
    """
    pure mathemtical optimisation of lin. equation of model
    """

    # maximize z = x + 2y <--> minimize -z = -x - 2y

    # coefficients
    obj = [-x for x in list(lin_reg.coef_)]
    obj.append(-1) # for intercept

    # boundaries for variables
    bnd = [(0, 1) for x in list(lin_reg.coef_)]
    bnd.append((lin_reg.intercept_,lin_reg.intercept_)) # for intercept

    # run optimisation
    opt = linprog(c=obj, bounds=bnd, method="revised simplex")
    print("---> optimal solution, value:", opt.x, opt.fun)

def write_resulting_parameters(data, kwargs, savename="results.csv"):
    """
    writes a csv file with our model and optimisation results
    """
    with open(savename, "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['Name','Koeffizient',"Optimaler Wert"])
        for mytuple in data:
            csv_writer.writerow(mytuple)
        csv_writer.writerow(["","",""])
        csv_writer.writerow(["","",""])
        csv_writer.writerow(["","",""])
        for mytuple in kwargs:
            csv_writer.writerow(mytuple)

def get_optimal_values(df: pd.DataFrame, dic_coeff: dict={}) -> list:
    """
    this runs the optimisation by hand
    max. target -> maximise function, negative coefficients have negative impact on winpercent -> exclude them, otherwise include
    1 if positive and binary, 0 if negative and binary, continuous unchanged (?)
    """
    binary_cells = [col for col in df if np.isin(df[col].dropna().unique(), [0, 1]).all()]
    rounded_coeff = []
    for c in dic_coeff:
        if dic_coeff[c] >= 0 and c in binary_cells:
            x = 1
        elif dic_coeff[c] < 0 and c in binary_cells:
            x = 0
        elif dic_coeff[c] < 0:
            x = 0
        else:
            x = 1#dic_coeff[c] (see comment above)
        rounded_coeff.append(x)
    return rounded_coeff

def stacked_histograms(df: pd.DataFrame, savename="stacked_vs_"):
    """
    creates (multiple) stacked histogram(s)
    """
    variables = ["pricepercent","winpercent","sugarpercent"]
    for var in variables:
        fig = plt.figure(figsize=(8, 8))
        df1 = df[df["chocolate"]==1]  # only choco
        df2 = df[df["fruity"]==1] # only fruity
        df3 = df[df["fruity"]*df["chocolate"]==1] # both choco and fruity
        df4 = df[df["fruity"]*df["chocolate"]==0] # neither choco or fruity
        df5 = df[df["nougat"] == 1] #nougat
        df6 = df[df["caramel"] == 1] #caramel
        df7 = df[df["peanutyalmondy"] == 1] #peanutyalmondy
        df8 = df[df["bar"] == 1] #bar
        df9 = df[df["hard"] == 1] # hard
        df10 = df[df["crispedricewafer"]==1] #crispedricewafer
        labels = ["choco", "fruity", "fruity && choco", "neither fruity nor choco", "nougat", "caramel", "peanut/almondy", "bar", "hard", "crispedricewafer"]
        ax = plt.hist([df1[var],df2[var],df3[var],df4[var], df5[var], df6[var], df7[var], df8[var], df9[var]],stacked=True)
        plt.legend(labels,frameon=False)
        plt.ylabel("Anzahl")
        if var == "pricepercent":
            plt.xlabel("Preisperzentil [%]")
        elif var == "winpercent":
            plt.xlabel("Gewinn [%]")
        elif var == "sugarpercent":
            plt.xlabel("Zuckerperzentil [%]")
        else:
            plt.xlabel(var)
        plt.tight_layout()
        plt.savefig(savename + f"{var}.png")

def create_boxplots(df: pd.DataFrame, savename: str="boxplot"):
    """
    calls boxplot creation
    """
    ytitle = "Gewinn [%]"
    make_boxplot(df=df,x="chocolate", y="winpercent", xtitle="Enthält Schokolade", ytitle=ytitle, savename=savename)
    make_boxplot(df=df,x="fruity", y="winpercent", xtitle="Enthält Frucht", ytitle=ytitle, savename=savename)
    make_boxplot(df=df,x="peanutyalmondy", y="winpercent", xtitle="Enthält Erdnuss/Mandel", ytitle=ytitle, savename=savename)
    make_boxplot(df=df,x="nougat", y="winpercent", xtitle="Enthält Nougat", ytitle=ytitle, savename=savename)
    make_boxplot(df=df,x="caramel", y="winpercent", xtitle="Enthält Nougat", ytitle=ytitle, savename=savename)
    make_boxplot(df=df,x="hard", y="winpercent", xtitle="Ist Bonbon (fest)", ytitle=ytitle, savename=savename)
    make_boxplot(df=df,x="bar", y="winpercent", xtitle="Ist Riegel", ytitle=ytitle, savename=savename)
    make_boxplot(df=df,x="pluribus", y="winpercent", xtitle="In Box o.ä.", ytitle=ytitle, savename=savename)
    make_boxplot(df=df,x="crispedricewafer", y="winpercent", xtitle="Enthält Reis/Keks o.ä.", ytitle=ytitle, savename=savename)

def make_boxplot(df: pd.DataFrame, x: str, y: str, xtitle: str, ytitle: str, savename: str="boxplot"):
    """
    creates a single boxplot
    """
    fig = plt.figure()
    ax = sns.boxplot(data=df, x=x, y=y, notch=False, showcaps=False, flierprops={"marker": "x"}, boxprops={"facecolor": (.4, .6, .8, .5)}, medianprops={"color": "coral"})
    ax.set_ylabel(ytitle)
    ax.set_xlabel(xtitle)
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[0] = 'Nein'
    labels[1] = 'Ja'
    ax.set_xticklabels(labels)
    plt.savefig(f"{savename}_{y}_{x}.png")

def corr_plot(df: pd.DataFrame, savename: str='heatmap', show_lower_half_only: bool=True) -> None:
    """
    create a correlation plot of all columns in dataframe
    """
    f = plt.figure(figsize=(15, 10))
    matrix = df.corr().round(2)
    mask = None
    if show_lower_half_only:
        mask = np.triu(np.ones_like(matrix, dtype=bool))
    sns.heatmap(matrix, annot=True, vmax=1, vmin=-1, center=0, cmap="vlag", mask=mask)
    plt.title('Correlation Matrix', fontsize=12);
    plt.tight_layout()
    plt.savefig(savename + ".png")

def scatter_plot(df: pd.DataFrame, savename: str="scatterplot") -> None:
    """
    simple scatter plot with two components
    """
    plt.figure(figsize=(15,5))
    cond = df.fruity > 0
    subset_a = df[cond]
    subset_b = df[~cond]
    plt.scatter(subset_a.winpercent, subset_a.chocolate, s=90, c='b', label='fruity')
    plt.scatter(subset_b.winpercent, subset_b.chocolate, s=60, c='r', label='not fruity')
    plt.xlabel("Gewinn [%]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(savename + ".png")

def scatter_plot_win_price(df: pd.DataFrame, savename: str="scatterplot_win_price") -> None:
    """
    simple scatter plot with two components
    """
    plt.figure(figsize=(15,15))
    subset_a = df["winpercent"]
    subset_b = df["pricepercent"]
    plt.scatter(subset_a, subset_b, s=90)
    plt.xlabel("Gewinn [%]")
    plt.ylabel("Preisperzentil [%]")
    plt.tight_layout()
    plt.savefig(savename + ".png")

def scatter_plot_win_sugar(df: pd.DataFrame, savename: str="scatterplot_win_sugar") -> None:
    """
    simple scatter plot with two components
    """
    plt.figure(figsize=(15,15))
    subset_a = df["winpercent"]
    subset_b = df["sugarpercent"]
    plt.scatter(subset_a, subset_b, s=90)
    plt.xlabel("Gewinn [%]")
    plt.ylabel("Zuckerperzentil [%]")
    plt.tight_layout()
    plt.savefig(savename + ".png")

def bar_plot(df: pd.DataFrame, x: str="winpercent", y: str="competitorname", sort_by: str="winpercent", title: str="", savename: str="barplot") -> None:
    """
    create a bar plot for best performing candies
    """
    df2 = df.sort_values(by=sort_by, ascending=False)
    plt.figure(figsize=(15,15))

    # attention: manually reduce the data
    minimal_winpercent_value = 0.67
    df2 = df2.drop(df[df.winpercent < minimal_winpercent_value].index)

    sns.set(style="ticks",font_scale=2)
    sns.set_style("ticks",{'axes.grid' : True})
    ax = sns.barplot(y=df2[y], x=df2[x])
    ax.set(xlim=[minimal_winpercent_value-0.02, 0.85])
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    labels = ["choco" if x==1 else "no choco" for x in df2['chocolate'].tolist()]
    ax.bar_label(ax.containers[0], labels=labels, padding=3, fontsize=15)
    ax.set_xlabel('Gewinn [%]', labelpad=20)
    ax.set_ylabel('Süßigkeit', labelpad=20)
    ax.set_title('Beliebteste Süßigkeiten')

    plt.title(title)
    plt.tight_layout()
    plt.savefig(savename + ".png")

def create_outputdir(path: str="test"):
    if not os.path.exists(path):
       os.makedirs(path)

if __name__ == "__main__":
    main()

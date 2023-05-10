"""
time series analysis:

main resource: https://towardsdatascience.com/the-complete-guide-to-time-series-analysis-and-forecasting-70d476bfe775
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
plt.style.use('seaborn')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
from scipy.optimize import minimize
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from tqdm import tqdm_notebook
from itertools import product

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

import warnings
warnings.filterwarnings('ignore')



def moving_average(data):
    if 0:
        plt.figure(figsize=(17, 8))
        plt.plot(data["Buchungsdatum (BL)_datetime"], data.ST)
        plt.title('MERCEDES ST')
        plt.ylabel('Number of entries')
        plt.xlabel('Time')
        plt.grid(True)
        plt.show()




    def plot_moving_average(data, window, plot_intervals=False, scale=1.96):

        series = data.ST
        x = data["Buchungsdatum (BL)_datetime"]
        print("series: ", series)
        rolling_mean = series.rolling(window=window, min_periods=window).mean()
        print("rolling mean: ", rolling_mean)

        plt.figure(figsize=(17,8))
        plt.title('Moving average\n window size = {}'.format(window))
        plt.plot(rolling_mean, 'g', label='Rolling mean trend')

        #Plot confidence intervals for smoothed values
        if plot_intervals:
            mae = mean_absolute_error(series[window:], rolling_mean[window:])
            deviation = np.std(series[window:] - rolling_mean[window:])
            lower_bound = rolling_mean - (mae + scale * deviation)
            upper_bound = rolling_mean + (mae + scale * deviation)
            plt.plot(upper_bound, 'r--', label='Upper bound / Lower bound')
            plt.plot(lower_bound, 'r--')


        plt.plot(series[window:], label='Actual values')
        plt.legend(loc='best')
        plt.grid(True)
        plt.title('MERCEDES ST')
        plt.ylabel('Number of entries')
        plt.xlabel('Time')

        counter = 0
        for vl in range(0,72):
            if vl % 12 == 0:
                plt.axvline(x=vl, color = 'black', linestyle = '--')
                year = 2017 + counter
                plt.text(vl+0.2,0,str(year),rotation=90)
                counter += 1


        plt.savefig("rolling_mean" + str(window) + ".png")

    #Smooth by the previous 5 days (by week)
    plot_moving_average(data, 6, plot_intervals=True)

    #Smooth by the previous month (30 days)
    #plot_moving_average(data, 30)

    #Smooth by previous quarter (90 days)
    #plot_moving_average(data, 90, plot_intervals=False)


def exp_smoothing(data):
    def exponential_smoothing(series, alpha):

        result = [series[0]] # first value is same as series
        for n in range(1, len(series)):
            result.append(alpha * series[n] + (1 - alpha) * result[n-1])
        return result

    def plot_exponential_smoothing(series, alphas):

        plt.figure(figsize=(17, 8))
        for alpha in alphas:
            plt.plot(exponential_smoothing(series, alpha), label="Alpha {}".format(alpha))
        plt.plot(series.values, "c", label = "Actual")
        plt.legend(loc="best")
        plt.axis('tight')
        plt.title("Exponential Smoothing")
        plt.grid(True);
        plt.savefig("exp_smoothgin" + ".png")

    plot_exponential_smoothing(data["ST"], [0.05, 0.3])


def double_exp_smoothing(data):

    def double_exponential_smoothing(series, alpha, beta):

        result = [series[0]]
        for n in range(1, len(series)+1):
            if n == 1:
                level, trend = series[0], series[1] - series[0]
            if n >= len(series): # forecasting
                value = result[-1]
            else:
                value = series[n]
            last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
            trend = beta * (level - last_level) + (1 - beta) * trend
            result.append(level + trend)
        return result

    def plot_double_exponential_smoothing(series, alphas, betas):

        plt.figure(figsize=(17, 8))
        for alpha in alphas:
            for beta in betas:
                plt.plot(double_exponential_smoothing(series, alpha, beta), label="Alpha {}, beta {}".format(alpha, beta))
        plt.plot(series.values, label = "Actual")
        plt.legend(loc="best")
        plt.axis('tight')
        plt.title("Double Exponential Smoothing")
        plt.grid(True)
        plt.savefig("doubel_exp_smoothgin" + ".png")

    plot_double_exponential_smoothing(data["ST"], alphas=[0.9, 0.02], betas=[0.9, 0.02])


def dicky_fuller(data):

    def tsplot(y, lags=None, figsize=(12, 7), syle='bmh', name_addition=""):

        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        with plt.style.context(style='bmh'):
            fig = plt.figure(figsize=figsize)
            layout = (2,2)
            ts_ax = plt.subplot2grid(layout, (0,0), colspan=2)
            acf_ax = plt.subplot2grid(layout, (1,0))
            pacf_ax = plt.subplot2grid(layout, (1,1))

            y.plot(ax=ts_ax)
            p_value = sm.tsa.stattools.adfuller(y)[1]
            ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
            smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
            smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
            plt.tight_layout()
            plt.savefig("dicky_filler" + name_addition + ".png")

    tsplot(data["ST"], lags=12, name_addition="first")

    # Take the first difference to remove to make the process stationary
    data_diff = data["ST"] - data["ST"].shift(1)
    tsplot(data_diff[1:], lags=12, name_addition="second")


def sarima(data):
    #Set initial values and some bounds
    ps = range(0, 12)
    d = 1
    qs = range(0, 12)
    Ps = range(0, 12)
    D = 1
    Qs = range(0, 12)
    s = 12

    #Create a list with all possible combinations of parameters
    parameters = product(ps, qs, Ps, Qs)
    parameters_list = list(parameters)
    len(parameters_list)

    # Train many SARIMA models to find the best set of parameters
    def optimize_SARIMA(parameters_list, d, D, s):
        """
            Return dataframe with parameters and corresponding AIC

            parameters_list - list with (p, q, P, Q) tuples
            d - integration order
            D - seasonal integration order
            s - length of season
        """

        results = []
        best_aic = float('inf')

        for param in tqdm_notebook(parameters_list):
            try: model = sm.tsa.statespace.SARIMAX(data["ST"], order=(param[0], d, param[1]),
                                                   seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
            except:
                continue

            aic = model.aic

            #Save best model, AIC and parameters
            if aic < best_aic:
                best_model = model
                best_aic = aic
                best_param = param
            results.append([param, model.aic])

        result_table = pd.DataFrame(results)
        result_table.columns = ['parameters', 'aic']
        #Sort in ascending order, lower AIC is better
        result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)

        return result_table

    result_table = optimize_SARIMA(parameters_list, d, D, s)

    #Set parameters that give the lowest AIC (Akaike Information Criteria)
    p, q, P, Q = result_table.parameters[0]

    best_model = sm.tsa.statespace.SARIMAX(data["ST"], order=(p, d, q),
                                           seasonal_order=(P, D, Q, s)).fit(disp=-1)

    print(best_model.summary())


def main():
    #csv_file = "groupby_month.csv"
    #csv_file = "groupby_day.csv"
    csv_file = "groupby.csv"

    data = pd.read_csv(csv_file)



    moving_average(data)
    exp_smoothing(data)
    double_exp_smoothing(data)
    dicky_fuller(data)
    #sarima(data)

if __name__ == "__main__":
    main()

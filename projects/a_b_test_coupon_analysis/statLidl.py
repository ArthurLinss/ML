from main import Analysis 
import pandas as pd 
import matplotlib.pyplot as plt
plt.style.use('seaborn-dark-palette')
import numpy as np
from scipy import stats

def main():
    df = readData()
    merged_df = grouping(df)
    statTTest(merged_df,var="turnover_mean")
    barlpot(merged_df)


def calcValues(g):

    g.columns = ['year', 'week', 'turnover_sum', 'turnover_mean', 'profit_sum', 'profit_mean', "return_value_sum", "return_value_mean", "cancellation_value_sum", "cancellation_value_mean", 'warengruppe_percent', 'yearweek',"Auftragsnummer","Kundennr","coupon"]
    
    def get_ratio(unique_customers, unique_orders):
        return unique_customers / unique_orders

    
    g['avg_orders_per_customer'] = get_ratio(g['Auftragsnummer'],g['Kundennr'],)
    g['avg_turnover_per_customer'] = get_ratio(g['turnover_sum'],g['Kundennr'],)
    g['avg_turnover_per_order'] = get_ratio(g['turnover_sum'],g['Auftragsnummer'],)
    g['avg_cancellation_value_per_customer'] = get_ratio(g['cancellation_value_sum'],g['Kundennr'],)
    g['avg_return_value_per_customer'] = get_ratio(g['return_value_sum'],g['Kundennr'],)

    return g


def grouping(df):

    def warengruppe_percentages(series):
        percentage_counts = series.value_counts(normalize=True) * 100
        percentage_counts = percentage_counts.reindex([0, 1, 2], fill_value=0)
        return tuple(percentage_counts)

    def count_unique(series):
        return len(series.unique())
        
    def count_non_empty_strings(series):
        return (series != '').sum()

    agg_dic = {
        "turnover_sum":pd.NamedAgg(column='turnover', aggfunc='sum'),
        "turnover_mean":pd.NamedAgg(column='turnover', aggfunc='mean'),
        "profit_sum":pd.NamedAgg(column='profit', aggfunc='sum'),
        "profit_mean":pd.NamedAgg(column='profit', aggfunc='mean'),
        "retour_sum":pd.NamedAgg(column='return_value', aggfunc='sum'),
        "retour_mean":pd.NamedAgg(column='return_value', aggfunc='mean'),
        "storno_sum":pd.NamedAgg(column='cancellation_value', aggfunc='sum'),
        "storno_mean":pd.NamedAgg(column='cancellation_value', aggfunc='mean'),
        "warengruppe_percent":pd.NamedAgg(column='warengruppe', aggfunc=warengruppe_percentages),
        "yearweek":pd.NamedAgg(column='yearweek', aggfunc='first'),
        "Auftragsnummer":pd.NamedAgg(column='Auftragsnummer', aggfunc=count_unique),
        "Kundennr":pd.NamedAgg(column='Kundennr', aggfunc=count_unique),
        "coupons":pd.NamedAgg(column="coupon", aggfunc=count_non_empty_strings),
    }
    g = df.groupby(['year', 'week']).agg(**agg_dic).reset_index()

    g = calcValues(g)


    print(g)


    #df_action20 = df[df['coupon'] == "action20"]
    df_action20 = df.loc[(df['week'] == 42) & (df['coupon'] == 'action20')]
    #gf20 = df_action20.groupby(['year', 'week'], as_index=False).agg(agg_dic)
    gf20 = df_action20.groupby(['year', 'week']).agg(**agg_dic).reset_index()
    gf20["yearweek"] = "2020-42-c"
    gf20 = calcValues(gf20)
    #print(gf20)
    df_notaction20 = df.loc[(df['week'] == 42) & (df['coupon'] == "") & (df['year'] == 2020)]
    #gfnot20 = df_notaction20.groupby(['year', 'week'], as_index=False).agg(agg_dic)
    gfnot20 = df_notaction20.groupby(['year', 'week']).agg(**agg_dic).reset_index()
    gfnot20["yearweek"] = "2020-42-nc"
    gfnot20 = calcValues(gfnot20)
    #print(gfnot20)

    df_action19 = df.loc[(df['week'] == 42) & (df['coupon'] == 'action19')]
    #gf19 = df_action19.groupby(['year', 'week'], as_index=False).agg(agg_dic)
    gf19 = df_action19.groupby(['year', 'week']).agg(**agg_dic).reset_index()
    gf19["yearweek"] = "2019-42-c"
    gf19 = calcValues(gf19)
    #print(gf19)
    df_notaction19 = df.loc[(df['week'] == 42) & (df['coupon'] == "") & (df['year'] == 2019)]
    #gfnot19 = df_notaction19.groupby(['year', 'week'], as_index=False).agg(agg_dic)
    gfnot19 = df_notaction19.groupby(['year', 'week']).agg(**agg_dic).reset_index()
    gfnot19["yearweek"] = "2019-42-nc"
    gfnot19 = calcValues(gfnot19)
    #print(gfnot19)

    merged_df = pd.concat([g, gf19, gfnot19, gf20, gfnot20], ignore_index=True)
    merged_df = merged_df.sort_values(by=['year', 'week'])
    merged_df = merged_df.reset_index(drop=True)

    print("Modified Dataframe: \n")
    print(merged_df)


    return merged_df

def barlpot(df):
    merged_df = df
    barplot(variable="turnover_mean", df=merged_df)
    barplot(variable="cancellation_value_mean",df=merged_df)
    barplot(variable="return_value_mean",df=merged_df)
    barplot(variable="profit_mean",df=merged_df)

        
def dfToDic(df, key_column, value_column):
    """
    transforms padndas dataframe df to dicionary given two columns for key and value
    """
    dic = df.set_index(key_column)[value_column].to_dict()
    return dic
    

def runTTest(data_list, single_value, alpha = 0.05, name="saveaddition"):
    data = np.array(data_list)
    t_statistic, p_value = stats.ttest_1samp(data, single_value)
    print("\n\n\nrunning t-test for\n value: %s" % (single_value))
    print("data: ", data)
    sig_dev = False
    if p_value < alpha:
        print("The value deviates significantly from the set of values.")
        sig_dev = True
    else:
        print("The value does not deviate significantly from the set of values.")
        sig_dev = False

    print("name, t-statistic, p-value, alpha, sign. dev.")
    print(str(name) + ", " + str(t_statistic) + ", " + str(p_value) + ", " + str(alpha) + ", " + str(sig_dev))

    plt.figure()
    plt.hist(data, bins=10, alpha=0.7, density=True, label="Data")
    plt.axvline(x=single_value, color='black', linestyle='dashed', linewidth=1.5, label="KW 42")
    if 1:
        # Adding source information
        plt.text(
            0.98,
            -0.25,
            "p-value: %.2f" % p_value,
            transform=plt.gca().transAxes,
            fontsize=10,
            color='gray',
            ha='right'
        )
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel(name.split("-")[0])
    plt.ylabel("Number of entries per bin")
    plt.title("t-test" + " " + name.replace("_"," "))
    plt.legend()
    plt.tight_layout()
    plt.savefig("ttest" + name + ".png")

def statTTest(df, var):
    dict = dfToDic(df, "yearweek", var)
    all = dict["2020-42"]
    nc = dict["2020-42-nc"]
    c = dict["2020-42-c"]
    del dict["2020-42-nc"]
    del dict["2020-42"]
    del dict["2020-42-c"]

    # also delete 19 values
    try:
        del dict["2019-42-nc"]
        del dict["2019-42"]
        del dict["2019-42-c"]
    except:
        print("deletion of 2019 data in KW 42 not possible")

    values = list(dict.values())
    alpha = 0.05
    runTTest(values, all, alpha, "%s-KW42" % var)
    runTTest(values, nc, alpha, "%s-KW42nc" % var)
    runTTest(values, c, alpha, "%s-KW42c" % var)



def barplot(variable, df):
    turnover_dic = dfToDic(df, "yearweek", variable)
    plt.figure(figsize=(12, 6))
    specific_keys_colors = {'2019-42-c': 'green', '2020-42-c': 'green', '2019-42-nc': 'gray', '2020-42-nc': 'gray', '2019-42': 'orange', '2020-42': 'orange'}
    colors = [specific_keys_colors.get(key, "blue") for key in turnover_dic.keys()]
    plt.bar(turnover_dic.keys(), turnover_dic.values(), color=colors) #width=bar_width
    plt.xlabel('Week and Year')
    plt.ylabel('%s Mean' % variable)
    plt.xticks(rotation='vertical')
    plt.title('%s per week and year' % variable)
    plt.tight_layout()
    plt.savefig("barplot" + "_" + variable + ".png")

def readData():
    ana = Analysis()
    ana.turnover_cut = 500
    df = ana.prepareData()
    df = df.drop(["turnover","cancellation_value","return_value"],axis=1)
    df.rename(columns={"turnover2":"turnover", "cancellation_value2":"cancellation_value", "return_value2":"return_value"}, inplace=True)
    df = df.fillna('')
    verbose=False
    if verbose==True:
        print("Original Dataframe: \n")
        print(df.head())
        print(df.columns)
        print(df['coupon'].unique())
    return df 



if __name__ == "__main__":
    main()
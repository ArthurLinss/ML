import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn-dark-palette')




def main():
    ana = Analysis()
    ana.prepareData()
    #ana.means()
    #ana.totalValues()
    #ana.boxplot()
    ana.customerAnalysis()
    ana.inspectData()



def barPlot(xvals, yvals, xlabel, ylabel, title, savename, ylim=None):
    plt.figure(figsize=(10, 6))
    plt.bar(xvals, yvals)
    #plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=0)
    if ylim != None:
        plt.ylim(ylim)
    #plt.show()
    plt.savefig(savename + ".png")


class Analysis():

    def __init__(self):
        self.turnover_cut = 500

    def prepareData(self):
        f = "./Probeaufgabe_DataAnalyst_Datensatz_copy.csv"
        df = pd.read_csv(f,sep=";")
        # datetime
        df["datetime"] = pd.to_datetime(df["date_key"], format='%Y%m%d')
        df["weekyear"] = df["week"].astype(str) + "-" + df["year"].astype(str)
        df["yearweek"] = df["year"].astype(str) + "-" + df["week"].astype(str)
        df = df.drop("date_key", axis=1)
        # format
        df['turnover2'] = df['turnover'].str.replace(',', '.').astype(float)
        df['cancellation_value2'] = df['cancellation_value'].str.replace(',', '.').astype(float)
        df['return_value2'] = df['return_value'].str.replace(',', '.').astype(float)
        # calc
        df['profit'] = df["turnover2"] - df["cancellation_value2"] - df["return_value2"]

        # filter
        print("reading data and applying turnover cut at %s" % self.turnover_cut)
        dfcut = df[df["turnover2"]<self.turnover_cut]

        # check cut 
        dfremoved = df[df["turnover2"]>=self.turnover_cut]
        print("original number of rows: ", len(df.index))
        print("removed rows: ", len(dfremoved.index))
        print("remaining rows: ", len(dfcut.index))

        # dump to csv
        dfcut.to_csv("data.csv")

        return dfcut


    def inspectData(self):

        df = pd.read_csv("data.csv")

        print(df.columns)
        print("Warengruppen: ", df["warengruppe"].unique(),  df["warengruppe"].nunique())
        print("Anzahl Kunden: ", df["Kundennr"].nunique())
        print("Wochen mit Jahr: ", df["weekyear"].unique(), df["weekyear"].nunique())
        print("Wochen im jeweiligen Jahr: ")
        print(df.groupby("year")["week"].nunique())
        print("Kunden im jeweiligen Jahr: ")
        print(df.groupby("year")["Kundennr"].nunique())
        print("Bestellungen im Jahr: ")
        print(df.groupby("year")["Auftragsnummer"].nunique())
        print("Umsatz im jeweiligen Jahr: ")
        print(df.groupby("year")["turnover2"].sum())
        print("Profit im jeweiligen Jahr: ")
        print(df.groupby("year")["profit"].sum())
        print("Durchschnittlicher Umsatzwert in der jeweiligen Woche: ")
        average_turnover_per_week = df.groupby(['week'])['turnover2'].mean()#.reset_index()
        print(average_turnover_per_week)
        print("Durchschnittlicher Umsatzwert pro Woche pro Bestellung: ")
        average_turnover_per_week_per_sell = df.groupby(['week','Auftragsnummer'])['turnover2'].mean()#.reset_index()
        print(average_turnover_per_week_per_sell)

        print("durchschnittliche anzahl bestellungen pro kunde pro woche")
        average_orders_per_week_year = df.groupby(['yearweek', 'Kundennr'])['Auftragsnummer'].nunique().reset_index()
        # Gruppiere nach Jahr und Woche, und berechne den Durchschnitt der Bestellanzahl pro Gruppe
        average_orders_per_week_year = average_orders_per_week_year.groupby(['yearweek'])['Auftragsnummer'].mean().reset_index()
        print(average_orders_per_week_year)
        barPlot(xvals=average_orders_per_week_year.yearweek, yvals=average_orders_per_week_year.Auftragsnummer, xlabel="Year and Week", ylabel='Avergage Number of Orders per Customers', title='',savename="average_orders_per_week_year", ylim=[1.0,1.01])




    def customerAnalysis(self):
        """
        analyse customer
        - how many in total
        - how many new in week 42
        """
        df = pd.read_csv("data.csv")

        # Kunden insgesamt
        customer_total = df['Kundennr'].nunique()
        print("customer total:\n", customer_total)

        # Kunden pro Woche
        customers_per_week = df.groupby('yearweek')['Kundennr'].nunique()
        print("Anzahl Kunden pro Woche:\n", customers_per_week)
        barPlot(xvals=customers_per_week.index, yvals=customers_per_week.values, xlabel="Year and Week", ylabel='Number of Customers', title='Total Number of unique Customers per Week',savename="totalNumberOfCustomers")


        orders_per_week = df.groupby('yearweek')['Auftragsnummer'].nunique()
        print("Anzahl Orders pro Woche:\n", orders_per_week)
        barPlot(xvals=orders_per_week.index, yvals=orders_per_week.values, xlabel="Year and Week", ylabel='Number of Orders', title='Total Number of Orders per Week',savename="totalNumberOfOrders")



        # Kunden, die in der jeweiligen Woche neu hinzugekommen sind

        def newCustomersPerWeek2(df):
                kundendaten = df
                # Sorting data by 'Kundennr' and 'week'
                kundendaten = kundendaten.sort_values(by=['Kundennr', 'week'])
                # Creating a set of unique customers per week
                unique_customers_per_week = kundendaten.groupby('week')['Kundennr'].apply(set)
                # Calculate the number of new customers per week, which did not bought something in the week before
                new_customers_per_week = []
                all_customers = set()
                for week, customers in unique_customers_per_week.items():
                    new_customers = customers - all_customers
                    new_customers_per_week.append(len(new_customers))
                    all_customers.update(customers)
                # Create a DataFrame with all weeks and the corresponding new customer counts
                all_weeks = range(min(kundendaten['week']), max(kundendaten['week']) + 1)
                result = pd.DataFrame({'week': all_weeks, 'new_customers': new_customers_per_week})
                print("Anzahl neu hinzugekommener Kunden pro Woche:\n", result)
                return result

        new_customers_per_week = newCustomersPerWeek2(df)

        print("Did customers bought more than twice? How often?\n")
        customer_purchase_counts = df['Kundennr'].value_counts()
        print(customer_purchase_counts.head(10))

        def firstWeekOfPurchase(df):
                """
                get first week of purchase of customers
                """
                kundendaten = df
                # Group by 'Kundennr' and find the first week of purchase for each customer
                first_week_of_purchase = kundendaten.groupby('Kundennr')['week'].min()
                print(first_week_of_purchase)
                # Print the result including all customers with multiple purchases
                print("First week of purchase for customers with multiple purchases:\n", first_week_of_purchase)
                # Plot the data in a bar plot
                plt.figure(figsize=(10, 6))
                first_week_of_purchase.value_counts().sort_index().plot(kind='bar')
                #plt.grid(True)
                plt.xlabel('Week')
                plt.ylabel('Number of Customers')
                plt.title('Number of Customers with First Purchase in Each Week')
                plt.xticks(rotation=0)
                #plt.show()
                plt.savefig("numberOfCustomersFirstPurchasePerWeek.png")
                return first_week_of_purchase

        firstWeekOfPurchase(df)

        def firstWeekOfPurchaseMorethanTwice(df):
                """
                get first week of purchase of customers who bought more than one time (at least two times)
                """
                kundendaten = df



                # Calculate the number of purchases per customer
                purchase_counts = kundendaten.groupby('Kundennr')['week'].count()

                # Filter customers who bought at least twice
                multiple_purchase_customers = purchase_counts[purchase_counts >= 2]

                # Get the first week of purchase for each customer who bought at least twice
                first_week_of_purchase = kundendaten[kundendaten['Kundennr'].isin(multiple_purchase_customers.index)]
                first_week_of_purchase = first_week_of_purchase.groupby('Kundennr')['week'].min()

                print("First week of purchase for customers who bought more than once:\n", first_week_of_purchase)

                # Plot the data in a bar plot
                plt.figure(figsize=(10, 6))
                first_week_of_purchase.value_counts().sort_index().plot(kind='bar')
                plt.xlabel('Week')
                plt.ylabel('Number of Customers')
                plt.title('Number of Customers with First Purchase in Each Week (Bought More Than Once)')
                plt.xticks(rotation=0)
                #plt.show()
                plt.savefig("numberOfCustomersFirstPurchasePerWeekMoreThanOnce.png")

                return first_week_of_purchase

        firstWeekOfPurchaseMorethanTwice(df)

    def printKPI(self, df, col):
        mean = df[col].mean()
        median = df[col].median()
        std = df[col].std()
        print(" Column: %s\n" % col, "Mean: %.2f \n" % mean, "Median: %.2f \n" % median, "StdDev: %2.f\n" % std)
        return mean, median, std


    def boxplot(self):
        df = pd.read_csv("data.csv")
        # Erstellen eines Boxplots von der Spalte "turnover"
        plt.figure(figsize=(8, 6))
        plt.boxplot(df['turnover2'], vert=True)  # vert=False fuer horizontalen Boxplot
        plt.title('Boxplot of Turnover')
        plt.xlabel('Turnover')
        plt.ylabel('')
        plt.yscale('log')  # Setzen der logarithmischen x-Achse

        plt.show()

    def totalValues(self):
        year=2019
        week=42
        if year==2019:
            action = "19"
        if year==2020:
            action="20"

        df = pd.read_csv("data.csv")
        df19 = df[df["year"]==year]
        #turnover_during_action = df19[(df19['week'] == week) & (df19['coupon'] == 'action%s' % action)]['turnover2'].sum()
        #profit_during_action = df19[(df19['week'] == week) & (df19['coupon'] == 'action%s' % action) ]['profit'].sum()

        turnover_during_action = df19[(df19['week'] == week)]['turnover2'].sum()
        profit_during_action = df19[(df19['week'] == week)]['profit'].sum()

        print("Turnover sum:", turnover_during_action)
        print("Profit sum:", profit_during_action)


        # Beispielcode zur Berechnung des durchschnittlichen Bestellwerts waehrend der Aktion
        average_order_value_during_action = turnover_during_action / df[(df['week'] == week) & (df['coupon'] == 'action%s' % action)]['Auftragsnummer'].nunique()

        print("Average order value:", average_order_value_during_action)



    def means(self):
        """
        analysing shift of means, e.g. of turnover
        """
        df = pd.read_csv("data.csv")
        df.drop(["turnover","cancellation_value","return_value"],axis=1)
        cols = df["weekyear"].unique()
        dfs = {}
        for col in cols:
            dfs[col] = df[df["weekyear"]==col]

        kw42_19_action_data = df[(df['coupon'] == 'action19') & (df['weekyear'] == "42-2019")]
        dfs["42-2019-action"] = kw42_19_action_data
        kw42_19_other_data = df[(df['coupon'] != 'action19') & (df['weekyear'] == "42-2019")]
        dfs["42-2019-noaction"] = kw42_19_other_data
        kw42_20_action_data = df[(df['coupon'] == 'action20') & (df['weekyear'] == "42-2020")]
        dfs["42-2020-action"] = kw42_20_action_data
        kw42_20_other_data = df[(df['coupon'] != 'action20') & (df['weekyear'] == "42-2020")]
        dfs["42-2020-noaction"] = kw42_20_other_data


        values = ["turnover2","cancellation_value2","return_value2","profit"]
        for value in values:

            categories = []
            mean_values= []
            median_values = []
            for name, data in dfs.items():
                print(name)
                print(data.head(5))
                mean, median, std = self.printKPI(data,value)
                categories.append(name)
                mean_values.append(mean)
                median_values.append(median)


            fig, ax = plt.subplots()
            plt.style.use('ggplot')

            # Set positions for the bars
            bar_width = 0.4
            indices = range(len(categories))

            # Create bars for mean values
            ax.barh(indices, mean_values, bar_width, label='Mean', color="blue")

            mean_of_means = sum(mean_values) / len(mean_values)
            ax.axvline(x=mean_of_means, color='blue', linestyle='--', label='Mean of Means', linewidth=0.7)
            mean_of_medians = sum(median_values) / len(median_values)
            ax.axvline(x=mean_of_medians, color='orange', linestyle='--', label='Mean of Medians', linewidth=0.7)


            # Create bars for median values, shift the x positions
            ax.barh([index + bar_width for index in indices], median_values, bar_width, label='Median', color="orange")

            ax.set_yticks([index + bar_width / 2 for index in indices])
            ax.set_yticklabels(categories)
            ax.invert_yaxis()  # Invert y-axis to have Category 1 at the top

            ax.set_xlabel('Values')
            ax.set_title('Mean and Median Comparison of %s' % value)

            ax.legend()

            plt.tight_layout()
            plt.savefig("barplot_mean_median_%s.png" % value)



if __name__ == "__main__":
    main()

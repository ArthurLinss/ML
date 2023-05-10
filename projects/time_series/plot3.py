from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime, timedelta



class Plotting():
    """
    facility to generace nice plots fast
    """

    def __init__(self):
        self.csv = ""
        self.x = ""
        self.y = ""
        self.t = ""
        self.ytitle = ""
        self.xtitle = ""

        self.add_vlines_per_year = True

    def plot(self):
        # Plot closing price
        data = pd.read_csv(self.csv)
        plt.figure(figsize=(17, 8))
        plt.plot(data[self.x], data[self.y])
        plt.title(self.t)
        plt.ylabel(self.ytitle)
        plt.xlabel(self.xtitle)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.grid(True)
        #plt.show()

        if self.add_vlines_per_year == True:
            counter = 0
            for vl in range(0,72):
                if vl % 12 == 0:
                    plt.axvline(x=vl, color = 'black', linestyle = '--')
                    year = 2017 + counter
                    #plt.text(vl+0.02,0.5,str(year),rotation=90)
                    counter += 1

        plt.savefig("plot" + "_" + self.csv.split(".")[-2] + ".png")



def main():
    p = Plotting()
    p.csv = "groupby.csv"
    p.x = "Buchungsdatum (BL)_datetime"
    p.y = "ST"
    p.t = "Mercedes Fallanalyse"
    p.ytitle = "Stückzahl"
    p.xtitle = "Zeit"

    p.plot()

if __name__ == "__main__":
    main()

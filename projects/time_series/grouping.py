import pandas as pd
from main import dateTransform
import pandas as pd
from matplotlib import pyplot as plt

df_BL = pd.read_csv("BL_2017-2022_cleaned.csv")
df_BL = dateTransform(df_BL)
date_column = 'Buchungsdatum (BL)_datetime'



s3 = df_BL.groupby([df_BL[date_column].dt.strftime('%Y-%m'),"Kundengruppe","Baureihe VT", "Umsatzgebiet"]).agg({'ST': 'sum'})#.reset_index()
s3df = s3.reset_index()
print(s3df)

s3df['Kundengruppe']=s3df['Kundengruppe'].astype('category').cat.codes
s3df['Baureihe VT']=s3df['Baureihe VT'].astype('category').cat.codes
s3df['Umsatzgebiet']=s3df['Umsatzgebiet'].astype('category').cat.codes
s3df[date_column]=s3df[date_column].astype('category').cat.codes

print(s3df)
corr = s3df.corr()
print(corr)


years = ["all"] #,"2017","2018","2019","2020","2021","2022"]


for startyear in years:
    if startyear != "all":
        df_BL = df_BL.loc[df_BL[date_column] >= "%s-01" % startyear]
        df_BL = df_BL.loc[df_BL[date_column] <= "%s-01"% (int(startyear)+1)]


    groups = ["Kundengruppe","Baureihe VT", "Umsatzgebiet"]
    for grouping in groups:
        print("nach %s" % grouping)

        s3 = df_BL.groupby([df_BL[date_column].dt.strftime('%Y-%m'), df_BL[grouping]]).agg({'ST': 'sum'})#.reset_index()
        s3df = s3.reset_index()


        s3df.to_csv(f"groupby_month_{grouping}_year_{startyear}.csv")
        print("s3: ", s3)

        s4 = df_BL.groupby([grouping]).agg({"ST" : "sum"})
        s4["ST [%]"] = s4["ST"]/s4["ST"].sum() * 100
        s4=s4.sort_values("ST [%]",ascending=False)
        print(s4)
        s3df.to_csv(f"groupby_month_percent_{grouping}_year_{startyear}.csv")
        print(s4["ST [%]"].sum())



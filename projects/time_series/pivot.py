#!/usr/bin/env python
# coding: utf-8

#  <h1 style="font-size:3rem;color:orange"> Jupyter Notebook Tutorial </h1> 

# # Python for beginner

# In[2]:


import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt


# In[47]:


csv_file = "BL_2017-2022_cleaned.csv"
df = pd.read_csv("/Users/arthur/Desktop/mercedes/case_study/" + csv_file)
df = df.drop(columns=["Bestelldatum","Fertig Ist","GLT-BAT-Ist","Übernahmedatum"])
df.head(5)
df["Buchungsdatum (BL)_datetime"] = pd.to_datetime(df["Buchungsdatum (BL)_datetime"], yearfirst=True, errors="coerce")
df.index = df["Buchungsdatum (BL)_datetime"]
df["ym"]=df["Buchungsdatum (BL)_datetime"].dt.strftime('%Y-%m')
print(df.dtypes)
print(df.columns)
print(df.head(5))


# In[54]:


def getPivot(df, group_name="Kundengruppe"):
    #g = df.groupby(by=[df.index.month, df.index.year,group_name]).agg({"ST":"sum"})
    #g = df.groupby(by=[[(df.index.year), (df.index.month)],group_name]).sum()
    #g = df.groupby([df.index.year, df.index.month, group_name]).sum()
    g = df.groupby([df["ym"], group_name]).agg({"ST":"sum"})#.sum()
    #g = g.pivot_table(index="Buchungsdatum (BL)_datetime",columns=group_name,values="ST")
    g = g.pivot_table(index="ym",columns=group_name,values="ST",fill_value=0,)
    return g


# In[55]:


g = getPivot(df, "Kundengruppe")
#g

# In[56]:


h = getPivot(df, "Baureihe VT")
#h


# In[57]:


i = getPivot(df,"Umsatzgebiet")
i
#

# In[76]:


dfst = df[["ST","ym"]].groupby("ym").sum()


# In[77]:


f = pd.concat([dfst, g, h, i], axis=1)


# In[78]:


print(f)


# In[13]:


f.to_csv("grouped_concat.csv")


# In[80]:


columns = f.columns


# In[90]:


print("[")
for x in columns:
    if x!="ST":
        print("\"" + x + "\"" +  ",")
print("]")


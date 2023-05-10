import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np


def sheetToDf(xls, sheet_name: str, drop_columns: list = []) -> pd.DataFrame:
    """
    reads a specific sheet in an excel file and return as dataframe
    """
    df = pd.read_excel(xls, sheet_name, skiprows=0)
    column_names = list(df)
    df = df.drop(drop_columns, axis=1)
    return df


def readDataframesFromExcelsheets(directory: str, csvPath: str = "./") -> list:
    """
    reads pandas dataframes from excel files
    also create csv file for each sheet (if csvPath!="-")
    """
    xls = pd.ExcelFile(directory)
    dfs = []
    for sheet_name in xls.sheet_names:
        df = sheetToDf(xls, sheet_name=sheet_name)
        df.name = sheet_name.replace(" ", "_")
        dfs.append(df)
        if csvPath != "-":
            df.to_csv(csvPath + sheet_name.replace(" ", "_") + ".csv", index=False)
    return dfs


def getData(directory: str, csvPath="./", force_excel=False) -> dict:
    """
    read (write) excel/csv files with the data and return dataframes of each excel sheet
    returns dictonary with sheetnames:dataframes
    """
    csv_files = list(filter(lambda f: f.endswith(".csv"), os.listdir(csvPath)))
    if len(csv_files) == 0 or force_excel:
        # the first time only (reading excel slow compared with csv)
        print("reading data from excel ...")
        dfs = readDataframesFromExcelsheets(directory=directory, csvPath=csvPath)
    else:
        print("reading data from csv ...")
        dfs = []
        for csvTable in csv_files:
            df = pd.read_csv(csvTable)
            df.name = csvTable.split(".")[-2]
            dfs.append(df)
    # create dictionary
    dic_dfs = {}
    for df in dfs:
        dic_dfs[df.name] = df
    return dic_dfs


def checkData(df):
    """
    checks the data frame for column content etc.
    """
    columns = df.columns.values
    print("cleaning/checking data for: " + df.name)
    print(df.head(5))
    print("columns: ", columns)
    with open("checkData_%s.txt" % df.name, "w", encoding="utf-8") as f:
        f.write("cleaning/checking data for: " + df.name)
        head = df.head(5).to_csv()
        f.write(head)
        f.write("columns: %s" % columns)

        for column in columns:
            vals = df[column].unique()
            print(
                "Column name: ",
                column,
                ", dtype: ",
                df[column].dtype,
                ", No. of different values: ",
                len(vals),
            )
            print("values: ", vals)
            print("\n")

            f.write(
                "Column name: "
                + column
                + ", dtype: "
                + str(df[column].dtype)
                + ", No. of different values: "
                + str(len(vals))
            )
            f.write("\n")


def cleanData(df):
    """
    clean and prepare data for further processing
    """
    # checkData(df)
    columns = df.columns.values
    if df.name == "AB":
        pass
    if df.name == "BL_2017-2022":
        #pass
        # TODO: check meaning of "X" and "#"
        rename = {
            "YES": "True",
            "NO": "False",
            "JA": "True",
            "NEIN": "False",
            "X": "True",
            "#":"False"
        }
        df["Verkauf aus Bestand"] = df["Verkauf aus Bestand"].replace(rename)

    df = dateTransform(df)
    # checkData(df)
    df.to_csv(df.name + "_cleaned" + ".csv")
    return df


def dateTransform(df: pd.DataFrame, replace: bool = False, errors="coerce"):
    """
    transforms date-like fields into datetime fields
    errors{‘ignore’, ‘raise’, ‘coerce’}, default ‘raise’
        If 'raise', then invalid parsing will raise an exception.
        If 'coerce', then invalid parsing will be set as NaT.
        If 'ignore', then invalid parsing will return the input.
    """
    renaming = ""
    if replace == False:
        renaming = "_datetime"
    AB_extradate_columns = ["GLT Err.", "Fertig Ist", "GLT-BAT-Err.", "GLT-BAT-Ist"]
    for column in df.columns:
        if "datum" in column or "Datum" in column or column in AB_extradate_columns:
            if column == "Bestelldatum" or column == "Buchungsdatum (BL)":
                df[column + renaming] = pd.to_datetime(
                    df[column], yearfirst=True, errors=errors
                )
            if (
                column == "Fertig ist"
                or column == "GLT Err."
                or column == "GLT-BAT-Ist"
                or column == "Fertig Ist"
                or column == "Übernahmedatum"
                or column == "GLT-BAT-Err."
            ):
                df[column + renaming] = pd.to_datetime(
                    df[column], dayfirst=True, errors=errors
                )
    return df


def main():
    """
    main algorithm
    """
    directory = "../CaseStudy_Vertriebssteuerung_Anon copy.xlsx"
    dfs = getData(directory=directory, csvPath="./", force_excel=False)

    df_AB = dfs["AB"]
    df_BL = dfs["BL_2017-2022"]

    df_AB = cleanData(df_AB)
    df_BL = cleanData(df_BL)

    checkData(df_AB)
    checkData(df_BL)

    date_column = "Buchungsdatum (BL)_datetime"

    print("grouping by month")
    df_BL_red = df_BL[["ST", date_column]]
    s = df_BL_red.groupby(df_BL_red[date_column].dt.strftime("%Y-%m"))
    # s = df_BL_red.groupby(df_BL_red['Buchungsdatum (BL)_datetime'])
    s2 = s["ST"].sum()
    print(s2)
    s2.to_csv("groupby_month.csv")

    print("grouping by day")
    df_BL_red2 = df_BL[["ST", date_column]]
    d = df_BL_red2.groupby(df_BL_red2[date_column].dt.strftime("%Y-%m-%d"))
    # d = df_BL_red2.groupby(df_BL_red2['Buchungsdatum (BL)_datetime'])
    d2 = d["ST"].sum()
    print(d2)
    d2.to_csv("groupby_day.csv")



    print("grouping by month full")
    e = df_BL.groupby([df_BL[date_column].dt.strftime("%Y-%m-%d"),"Kundengruppe","Umsatzgebiet","Baureihe VT"]).agg({"ST":"sum"})
    # s = df_BL_red.groupby(df_BL_red['Buchungsdatum (BL)_datetime'])
    #e3 = e["ST"].sum()
    e3 = e.reset_index()
    print(e3)
    e3.to_csv("groupby_month_full.csv")

    print("grouping by day full")
    #df_BL["delta_t"] = df_BL["GLT-BAT-Ist_datetime"].sub(df_BL["Bestelldatum_datetime"]).dt.days
    df_BL["delta_t"] = df_BL[date_column].sub(df_BL["Bestelldatum_datetime"]).dt.days
    df_BL["ym"]=df_BL["Buchungsdatum (BL)_datetime"].dt.strftime('%Y-%m')
    df_BL["m"]=df_BL["Buchungsdatum (BL)_datetime"].dt.strftime('%m')
    #f = df_BL.groupby([df_BL[date_column].dt.strftime("%Y-%m-%d"),"ym","delta_t","Kundengruppe","Umsatzgebiet","Baureihe VT"]).agg({"ST":"sum"})
    #f3 = f.reset_index()
    f3 = df_BL
    print(f3)
    f3.to_csv("groupby_day_full.csv")

if __name__ == "__main__":
    main()

import os
import tarfile
from six.moves import urllib

def fetchData(url: str=None, path: str=None):
    """
    create local directory for the output given path argument
    download data given as tar ball given a url argument
    extract tar ball
    """
    if not os.path.isdir(path):
        os.makedirs(path)
    tgzPath = os.path.join(path, url.split("/")[-1])
    urllib.request.urlretrieve(url, tgzPath)
    tgz = tarfile.open(tgzPath)
    tgz.extractall(path=path)
    tgz.close()
    print("fetching data succesful")

import pandas as pd
import pyspark.pandas as ps
import gzip

def loadDataAsDF(pathToFile: str=None, spark=True):
    """
    load csv file as pandas dataframe given a local path and a filename
    bool spark enables to return a pyspark dataframe instead of a pandas dataframe
    pyspark requires java to work
    """
    print(f"load data as dataframe from {pathToFile}")
    try:
        df = pd.read_csv(pathToFile)
    except:
        with gzip.open(pathToFile, 'rt', encoding='utf-8') as f:
            df = pd.read_csv(f)
    if spark:
        sparkDF = ps.from_pandas(df)
        df = sparkDF
    return df



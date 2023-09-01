from dataOrganizer import fetchData, loadDataAsDF

def readingData():
    verbose = True
    # define path for download
    ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    PATH = "datasets/housing"
    FILE = "/housing.tgz"
    URL = ROOT + PATH + FILE
    fetchData(url=URL, path=PATH)
    df = loadDataAsDF(pathToFile = PATH + FILE, spark=False)
    if verbose:
        print(df.head(10))
        df.info()
        print("Type: %s " % type(df))


def main():
    readingData()

if __name__ == "__main__":
    main()

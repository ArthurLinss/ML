from sklearn.datasets import make_blobs  # pip install scikit-learn
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import os.path
import random



class DBScan():
    def __init__(self):
        # pseudo data
        self.n_samples = 1000
        self.centers = [
            (3, 3),
            (6, 6),
            (1, 1),
        ]
        self.minP = 7
        self.epsilon = 0.3#1.0
        self.cluster_std = 0.5
        self.center_box = (0, 1)

    def genPseudoData(self):
        return make_blobs(
            n_samples=self.n_samples,
            centers=self.centers,
            n_features=len(self.centers),
            center_box=self.center_box,
            cluster_std=self.cluster_std,
        )

    def fitDBScan(self, data, epsilon, minP):
        db = DBSCAN(eps=epsilon, min_samples=minP)
        db.fit(data)
        self.getScanResult(db)
        self.plotScanResult(db=db, data=data)

    def genRandomColor(self):
        r = lambda: random.randint(0,255)
        return '#%02X%02X%02X' % (r(),r(),r())

    def getScanResult(self, db):
        labels = db.labels_
        #print(labels)
        res_n_clusters = len(np.unique(labels))
        print("Number of density-connected clusters: ", res_n_clusters)
        res_n_noisy_samples = np.sum(np.array(labels) == -1, axis=0)
        print("Number of noisy samples: ", res_n_noisy_samples)

    def plotScanResult(self, db, data):
        labels = db.labels_

        colors = {}
        for x in labels:
            colors[x] = self.genRandomColor()

        colz = list(map(lambda x: colors[x], labels))
        plt.scatter(data[:,0], data[:,1], c=colz, marker="o", picker=True)
        plt.title('Unsupervised DBScan Clustering Alogrithm')
        plt.xlabel('Axis data[0]')
        plt.ylabel('Axis data[1]')
        plt.savefig("scan_result.png")

    def run(self):
        data_dump = "./clusters.npy"
        if os.path.exists(data_dump):
            print("loading data ...")
            print("note: delete the file {data_dump} manually if you changed some model parameters")
            X = np.load(data_dump)
        else:
            print("generating data ...")
            X, y = self.genPseudoData()
            np.save(data_dump, X)

        print("fitting ...")
        scan = self.fitDBScan(X, self.epsilon, self.minP)


def main():
    dbs = DBScan()
    dbs.run()

if __name__ == "__main__":
    main()

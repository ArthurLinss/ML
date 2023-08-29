import scipy
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def t_test():
    """
    simple t-test example with random independent binomial distributions (with equal variance)
    tests for 0-hypothesis if 2 samples have identival average values (t-test quantifies the difference between the arithmetic means of the two samples)
    p-value quantifies the probability of observing as or more extreme values assuming the null hypothesis
    p-value larger than a chosen threshold (e.g. 5% or 1%, the signficance level) indicates that our observation is not so unlikely to have occurred by chance.
    """
    trials = 100 # number of trials
    probability = 0.9 # probability of each trial
    size = 100000
    b1 = np.random.binomial(n=trials, p=probability, size=size)
    b2 = np.random.binomial(n=trials, p=probability, size=size)


    counts_b1, bins_b1 = np.histogram(b1)
    plt.stairs(counts_b1, bins_b1)
    counts_b2, bins_b2 = np.histogram(b2)
    plt.stairs(counts_b2, bins_b2)
    plt.show()


    res = stats.ttest_ind(b1, b2)
    print(res)
    print("t-test statistic: ", res.statistic)
    print("p-value: ", res.pvalue)



def t_test_loop():
    trials = 10
    probability = 0.5
    size = [10,100,1000,5000,10000,20000,50000,100000,1000000,10000000]

    # binnomial equal normal if np>0.5 && n(1-p)>0.5

    p_vals_size = []
    for x in size:
        res = stats.ttest_ind(np.random.binomial(n=trials, p=probability, size=x), np.random.binomial(n=trials, p=probability, size=x))
        p_vals_size.append(res.pvalue)

    dic = {"pvals": p_vals_size, "samplesize": size}
    plt.plot('pvals', 'samplesize', data=dic)
    plt.yscale('log')
    plt.xlabel("p-value")
    plt.ylabel("sample size (binomial, n=%s, p=%s)" % (trials, probability))
    plt.show()

def main():
    #t_test()

    t_test_loop()

if __name__ == "__main__":
    main()

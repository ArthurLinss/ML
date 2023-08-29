"""
T-test for the means of two independent samples of scores
The p-value quantifies the probability of observing as or more extreme values assuming the null hypothesis
0-hypothesis: samples are drawn from populations with same population means
"""
import numpy as np
from scipy import stats

def t_test(population_a, population_b, alpha = 0.05):
    # Perform t-test to compare conversion rates
    t_stat, p_value = stats.ttest_ind(population_a, population_b)

    if p_value < alpha:
        # evidence against null hypothesis
        print("Statistically significant")
    else:
        # large p-value: observation is not so unlikely to have occurred by chance
        print("Not statistically significant, do not reject null hypothesis")

    # Print the t-statistic and p-value
    print("T-statistic:", t_stat)
    print("P-value:", p_value)


def main():
    # generate some pseudodata
    sample_size = 1000
    group_a = np.random.binomial(n=100, p=0.1, size=sample_size)
    group_b = np.random.binomial(n=100, p=0.12, size=sample_size)

    group_a_list = group_a.tolist
    group_b_list = group_b.tolist

    t_test(group_a,group_b)


if __name__ == "__main__":
    main()

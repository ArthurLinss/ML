


class myMetric():
    """
    implementations of accuarcy, precision and recall as metrics for model performances
    precision and recall are in tension typically
    """

    def __init__(self, tp, tn, fp, fn):
        self.tp = tp  # true positives
        self.tn = tn  # true negatives
        self.fp = fp  # false positives
        self.fn = fn  # false negatives

    @property
    def accuracy(self):
        """
        fraction of predictions model got right
        """
        acc = (self.tp + self.tn)/(self.tp + self.tn + self.fp + self.fn)
        return acc

    @property
    def precision(self):
        """
        proportion of positive identificatinos which were correct
        if there is no false positive, then precision=1
        """
        prec = self.tp / (self.tp + self.fp)
        return prec

    @property
    def recall(self):
        """
        proportion of actual positives which were identified correctly
        if there is no false negative, then recall = 1
        """
        rec = self.tp / (self.tp + self.fn)
        return rec


    def fscore(self, beta=1):
        """
        (weighted) harmonic mean of precision and recall
        beta=2 e.g. f2 score
        related to alpha and beta error, see https://en.wikipedia.org/wiki/F-score
        """
        f = (1+beta*beta) * (self.precision*self.recall) / ((beta*beta*self.precision) + self.recall)
        return f


    def tpr(self)
        """
        true positive rate
        synonym to recall
        y-axis for ROC curve (trp vs. fpr for different thresholds)
        """
        return self.recall

    def fpr(self):
        """
        false positive rate
        x-axis for ROC curve (trp vs. fpr for different thresholds)
        """
        return self.fp / (self.fp + self.tn)

import pytest
def test():


    metrics = myMetric(tp=7,tn=18,fp=1,fn=4)
    print("accuracy: ", metrics.accuracy)
    print("precision: ", metrics.precision)
    print("recall: ", metrics.recall)
    print("f1: ", metrics.fscore(beta=1))
    print("f2: ", metrics.fscore(beta=2))
    assert metrics.precision == pytest.approx(0.88, 0.01)
    assert metrics.recall == pytest.approx(0.64, 0.01)

    metrics = myMetric(tp=9,tn=16,fp=3,fn=2)
    print("accuracy: ", metrics.accuracy)
    print("precision: ", metrics.precision)
    print("recall: ", metrics.recall)
    print("f1: ", metrics.fscore(beta=1))
    print("f2: ", metrics.fscore(beta=2))
    assert metrics.precision == pytest.approx(0.75, 0.01)
    assert metrics.recall == pytest.approx(0.82, 0.01)


def main():
    test()


if __name__ == "__main__":
    main()

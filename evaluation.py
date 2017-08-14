from pandas import scatter_matrix
import pandas as pd
from itertools import combinations
from scipy.stats import norm, chi2_contingency
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier as DTC
import seaborn as sns
from datasets import split_data


def categorical(col):
    return str(col.dtype) == "category"


def limits(*args):
    return np.min(args), np.max(args)


def bins(limits, nbin=10):
    return np.linspace(*limits, num=nbin)


def bin_data(data, bins):
    return pd.cut(data, bins)


def encode(a, b, same_var=True):
    if same_var:
        a, b = a.cat.codes if categorical(a) else bin_data(a, bins(limits(a, b))).cat.codes, \
               b.cat.codes if categorical(b) else bin_data(b, bins(limits(a, b))).cat.codes
    else:
        a, b = a.cat.codes if categorical(a) else bin_data(a, bins(limits(a))).cat.codes, \
               b.cat.codes if categorical(b) else bin_data(b, bins(limits(b))).cat.codes
    return a, b


def cramers_corrected_v(a, b):
    contingency_table = pd.crosstab(a, b)
    chi2, p, df, ex = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    phi2 = chi2/n
    r, k = contingency_table.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / max(1, min((kcorr-1), (rcorr-1)))), p


def association_iid(original, sampled):
    return np.mean([cramers_corrected_v(*encode(original[col], sampled[col])) for col in original], axis=0)


def accuracy_iid(original, sampled):
    return np.mean([accuracy(*encode(original[col], sampled[col])) for col in original], axis=0)


def association(original, sampled):
    associations = []
    for var_a, var_b in combinations(original.columns, 2):
        expected_cv = cramers_corrected_v(*encode(original[var_a], original[var_b], False))
        sampled_cv = cramers_corrected_v(*encode(sampled[var_a], sampled[var_b], False))
        associations.append(1 - np.abs(expected_cv[0] - sampled_cv[0]))

    return np.mean(associations)


def accuracy(a, b):
    return 1-np.sum(np.abs((a.value_counts()/len(a)-b.value_counts()/len(b))))


def decision_tree_evaluation(xorig, yorig, xsamp, ysamp):
    xotrain, xotest, yotrain, yotest = split_data(xorig, yorig)

    clf1, clf2 = DTC(), DTC()
    clf1.fit(xotrain, yotrain)
    score1 = clf1.score(xotest, yotest)

    clf2.fit(xsamp, ysamp)
    score2 = clf2.score(xorig, yorig)

    print("\nClassification Accuracy for salary using decision trees:")
    print("Original : {:.4f}".format(score1))
    print("Sampled  : {:.4f}".format(score2))
    print("Ratio    : {:.4f}".format(score2/score1))

    return score1, score2


def report(original, sample):
    print("----- Results ------")
    print("Mean Accuracy   : {:.4f}".format(accuracy_iid(original, sample)))
    print("Mean Cramer's V : {:.4f} {:.4f}".format(*association_iid(original, sample)))
    print("Mean Inter-Variable Association : {:.4f}".format(association(original, sample)))



def plot_mnist_samples(vae):
    n = 15
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae._generator.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()


def compare_histograms(original, sample):
    f, ax = plt.subplots(2, original.shape[1])
    for i in range(original.shape[1]):
        ax[0, i].hist(original[:, i], normed=True)
        ax[1, i].hist(sample[:, i], normed=True)
        ax[1, i].set_xlim((0,1))
    plt.show()


def compare_matrix(original, sample):
    scatter_matrix(pd.DataFrame(original), alpha=0.4)
    scatter_matrix(pd.DataFrame(sample), alpha=0.4, marker="x")
    plt.show()

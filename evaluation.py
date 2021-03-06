from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import scatter_matrix
import os
from scipy.stats import norm, chi2_contingency
from sklearn.tree import DecisionTreeClassifier as DTC
from preprocessing.Vectorizer import Vectorizer

from preprocessing.datasets import split_data


def categorical(col):
    return str(col.dtype) == "category"


def limits(*args):
    return np.min(np.concatenate(args)), np.max(np.concatenate(args))


def bins(limits, nbin=10):
    return np.linspace(*limits, num=nbin)


def bin_data(data, bins):
    return pd.cut(data, bins, include_lowest=True)


def encode(a, b, same_var=True, lim=None):
    if same_var:
        a, b = a.cat.codes if categorical(a) else bin_data(a, bins(lim if lim else limits(a, b))).cat.codes, \
               b.cat.codes if categorical(b) else bin_data(b, bins(lim if lim else limits(a, b))).cat.codes
    else:
        a, b = a.cat.codes if categorical(a) else bin_data(a, bins(lim if lim else limits(a))).cat.codes, \
               b.cat.codes if categorical(b) else bin_data(b, bins(lim if lim else limits(b))).cat.codes
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


def association_iid(original, sampled, limits=None):
    return np.mean([cramers_corrected_v(*encode(original[col], sampled[col], lim=limits)) for col in original], axis=0)


def accuracy_iid(original, sampled, limits=None):
    return np.mean([accuracy(*encode(original[col], sampled[col], lim=limits)) for col in original], axis=0)


def correlation_difference(original, sampled, method="pearson"):
    m = pd.get_dummies(original).shape[1]
    corra = pd.get_dummies(original).corr(method=method)
    corrb = pd.get_dummies(sampled).corr(method=method)
    return 1 - np.nansum(np.tril((corra - corrb).abs())) / (m*(m-1)/2)


def association(original, sampled, limits=None):
    associations = []
    for var_a, var_b in combinations(original.columns, 2):
        expected_cv = cramers_corrected_v(*encode(original[var_a], original[var_b], False, limits))
        sampled_cv = cramers_corrected_v(*encode(sampled[var_a], sampled[var_b], False, limits))
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

    return score1, score2


def report(X, y, samples, vectorizer, model, params, dataset, binary, reorder):
    print("\n----- Results for dataset {} using {} ------".format(dataset, model))

    dec_samples = vectorizer.inverse_transform(samples, clip=vectorizer.feature_range)

    vec2 = Vectorizer()
    X_t = vec2.fit_transform(X, reorder=reorder)
    new = vec2.transform(dec_samples, apply_columns=False)

    acc = accuracy_iid(X, dec_samples, vectorizer.feature_range)
    print("Mean Accuracy   : {:.4f}".format(acc))

    v, p = association_iid(X, dec_samples, vectorizer.feature_range)
    print("Mean Cramer's V : {:.4f} {:.4f}".format(v, p))

    person_diff = correlation_difference(X, dec_samples)
    print("Pearson Correlation Difference : {:.4f}".format(person_diff))

    #spearman_diff = correlation_difference(X, dec_samples, "spearman")
    #print("Spearman Correlation Difference : {:.4f}".format(spearman_diff))

    if dataset in ["adult", "binary"]:
        if dataset == "adult":
            av = "salary"
        elif dataset == "binary":
            av = "15"

        iva = association(X, dec_samples, vectorizer.feature_range)
        print("Mean Inter-Variable Association : {:.4f}".format(iva))

        yaxis = [c for c in X_t.columns if av in c]
        print("Sampled class ratio : {:.4f}".format(np.ravel(new[yaxis]).sum()/len(new)))
        score1, score2 = decision_tree_evaluation(X_t.drop(yaxis, axis=1).as_matrix(), X_t[yaxis],
                                                  new.drop(yaxis, axis=1), new[yaxis])
        print("\nClassification Accuracy for {} using decision trees:".format(av))
        print("Original : {:.4f}".format(score1))
        print("Sampled  : {:.4f}".format(score2))
        print("Ratio    : {:.4f}".format(score2/score1))

        compare_histograms(X_t, new)
        if not os.path.exists("./results/figures"):
            os.makedirs("./results/figures")

        plt.savefig("./results/figures/histcomp_{}_{}_{}_{}".format(model, dataset, "binary" if binary else "cont",
                                                            "reordered" if reorder else "regular"))

        results = acc, v, p, person_diff, iva, score1, score2

    if dataset == "mnist":
        results = acc, v, p, person_diff, None, None, None

    save_results(dataset, model, params, reorder, binary, *results)
    return results


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
    cols = list(pd.DataFrame(original))
    original, sample = np.asarray(original), np.asarray(sample)
    f, ax = plt.subplots(2, original.shape[1], sharex=True, sharey=True, figsize=(20, 8))

    for i in range(original.shape[1]):
        ax[0, i].hist(original[:, i], normed=True, range=(0,1))
        ax[0, i].set_title(cols[i])
        ax[1, i].hist(sample[:, i], normed=True, range=(0,1))
        ax[1, i].set_ylim((0,10))
        ax[0, i].set_ylim((0,10))

    f.tight_layout()
    f.subplots_adjust(hspace=0)
    #plt.show()


def compare_matrix(original, sample):
    scatter_matrix(pd.DataFrame(original), alpha=0.4)
    scatter_matrix(pd.DataFrame(sample), alpha=0.4, marker="x")
    plt.show()


def save_results(*results):
    try:
        out = pd.read_excel("./results/out.xlsx")
    except:
        out = pd.DataFrame(columns=["dataset", "algorithm","params", "reordered", "binary", "accuracy", "cramers_v", "chi2p",
                                    "pearson", "iva", "dt_accuracy_original", "dt_accuracy_sampled"])
    out.loc[out.shape[0]] = list(results)
    if not os.path.exists("./results"):
        os.makedirs("./results")
    out.to_excel("./results/out.xlsx")

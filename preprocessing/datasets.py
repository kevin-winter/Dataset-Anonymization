from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston, fetch_california_housing, load_iris
import pandas as pd
import numpy as np
from keras.datasets import mnist
from itertools import product


def split_data(X, y):
    features_train, features_test, labels_train, labels_test = \
        train_test_split(X, y, test_size=0.2, stratify=y)
    try:
        return features_train.as_matrix(), features_test.as_matrix(), labels_train.as_matrix(), labels_test.as_matrix()
    except:
        try:
            return features_train.as_matrix(), features_test.as_matrix(), labels_train, labels_test
        except:
            return features_train, features_test, labels_train, labels_test

def adult_dataset(drop_y=True):
    dtypes = {"age": int,
              "workclass": "category",
              "fnlwgt": int,
              "education": "category",
              "education-num": int,
              "marital-status": "category",
              "occupation": "category",
              "relationship": "category",
              "race": "category",
              "sex": "category",
              "capital-gain": int,
              "capital-loss": int,
              "hours-per-week": int,
              "native-country": "category",
              "salary": "category"}

    X = pd.read_csv("./preprocessing/adult.txt", dtype=dtypes)
    y = X["salary"].cat.codes
    if drop_y:
        X.drop("salary", axis=1, inplace=True)
    return X, y


def mnist_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X = np.concatenate((x_train,x_test)).astype(float)/255
    X = X.reshape((len(X), np.prod(X.shape[1:])))
    y = np.concatenate((y_train,y_test))
    X = pd.DataFrame(X)
    return X, y


def binary_dataset(bits=16):
    X = pd.DataFrame(list(product(range(2), repeat=bits)), columns=[str(i) for i in range(bits)])
    y = X["15"]
    return X, y


def boston_houses_dataset():
    dataset = load_boston()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df['target'] = dataset.target
    return df, df['target']


from sklearn.model_selection import train_test_split
from keras.datasets import mnist
import pandas as pd
import numpy as np

from vectorize import Vectorizer
from VAE import VAE

def split_data(df, targets):
    features_train, features_test, labels_train, labels_test = \
        train_test_split(df, targets, test_size=0.1, stratify=targets)
    return features_train, features_test, labels_train, labels_test

def adult_dataset():
    headers = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship",
               "race","sex","capital-gain","capital-loss","hours-per-week","native-country","salary"]

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

    df = pd.read_csv("adult.txt", names=headers, dtype=dtypes)
    targets = df["salary"].cat.codes
    df.drop("salary", axis=1, inplace=True)

    vec = Vectorizer()
    dft = vec.transform(df)
    nrows = int(len(dft)/500) * 500
    return split_data(dft.as_matrix()[:nrows,:], targets[:nrows]), vec


def mnist_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    return (x_train, x_test, y_train, y_test), None


#(x_train, x_test, y_train, y_test), vec = adult_dataset()
#vae = VAE(original_dim=14, intermediate_dim=30, batch_size=10, latent_dim=4)


(x_train, x_test, y_train, y_test), vec = mnist_dataset()
vae = VAE(original_dim=784, intermediate_dim=256, batch_size=50, latent_dim=3)

vae.train(x_train, x_test)
vae.plot_embedding(x_test, y_test)
samples = vae.sample_z(100)
if vec is not None:
    print(vec.inverse_transform(samples))
#vae.plot_samples()
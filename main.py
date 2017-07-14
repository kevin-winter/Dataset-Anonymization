from VAE import VAE
from yadlt.models.boltzmann.dbn import DeepBeliefNetwork
from yadlt.utils import datasets, utilities
from datasets import adult_dataset, mnist_dataset, split_data
from sklearn.neural_network import BernoulliRBM
from sklearn.mixture import GaussianMixture
from pandas import scatter_matrix
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from Vectorizer import Vectorizer, from_dummies

def compare(original, sample):
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

dataset = "adult"
model = "gmm"
binary_encoding = False
reoder_categories = True

if dataset == "adult":
    X, y = adult_dataset()

elif dataset == "mnist":
    X, y = mnist_dataset()

vec = Vectorizer(binary=binary_encoding)
X_t = vec.fit_transform(X)
x_train, x_test, y_train, y_test = split_data(X_t, y)


if model == "gmm":
    gmm = GaussianMixture(n_components=5)
    gmm.fit(x_train, y_train)
    raw_samples = gmm.sample(500)
    if vec is not None:
        dec_samples = vec.inverse_transform(raw_samples[0])
    new = vec.transform(dec_samples)
    compare(x_train, new)

if model == "vae":
    vae = VAE(intermediate_dim=10, latent_dim=5, n_hiddenlayers=3)
    vae.train(x_train, x_test, epochs=50, batch_size=100, early_stopping=False)
    vae.plot_embedding(x_test, y_test)

    raw_samples = vae.sample_z(100)
    if vec is not None:
        dec_samples = vec.inverse_transform(raw_samples, decode=False)
    if dataset == "mnist":
        vae.plot_samples()
    if X.shape[1] < 20:
        compare(x_train, raw_samples)


if model == "dbn":
    pretrain = False
    trX, trY, vlX, vlY, teX, teY = datasets.load_mnist_dataset(mode='supervised')
    dbn = DeepBeliefNetwork(do_pretrain=pretrain, rbm_layers=[500, 10],
                        finetune_learning_rate=0.2, finetune_num_epochs=20)
    if pretrain:
        dbn.pretrain(trX, vlX)
    dbn.fit(trX, trY, vlX, vlY)
    print('Test set accuracy: {}'.format(dbn.score(teX, teY)))


if model == "rbm":
    rbm = BernoulliRBM(batch_size=100, verbose=1, n_iter=20)
    rbm.fit(x_train)

    n_samples = 20
    k = 100
    v = np.random.randint(2, size=(10, x_train.shape[1]))
    for i in range(k):
        v = rbm.gibbs(v)

    plt.figure(figsize=(10, 10))
    for x in v:
        plt.imshow(x.reshape(28,28), interpolation="none")
        plt.show()





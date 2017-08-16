from yadlt.models.boltzmann.dbn import DeepBeliefNetwork
from yadlt.utils import datasets, utilities
from sklearn.neural_network import BernoulliRBM
from sklearn.mixture import GaussianMixture

from datasets import adult_dataset, mnist_dataset, split_data, binary_dataset, boston_houses_dataset
from evaluation import *
from Vectorizer import Vectorizer
from SimpleGAN import SimpleGAN
from DCGAN import DCGAN
from VAE import VAE


def to_one_hot(y):
    if len(np.shape(y)) == 1:
        return np.eye(np.max(y)+1)[y]
    else:
        return y


dataset = "binary"
model = "rbm"
binary_encoding = False
reoder_categories = False


vec = Vectorizer(binary=binary_encoding)

if dataset == "adult":
    X, y = adult_dataset(drop_y=False)
    X_t = vec.fit_transform(X)

elif dataset == "mnist":
    X, y = mnist_dataset()
    X_t = vec.fit_transform(X)

elif dataset == "binary":
    X, y = binary_dataset()
    X_t = vec.fit_transform(X, True, False)

elif dataset == "boston":
    X, y = boston_houses_dataset()
    X_t = vec.fit_transform(X)



x_train, x_test, y_train, y_test = split_data(X_t, y)
n_samples = len(X)

if model == "gmm":
    gmm = GaussianMixture(n_components=12)
    gmm.fit(x_train)
    samples = gmm.sample(n_samples)[0]


if model == "vae":
    vae = VAE(intermediate_dim=40, latent_dim=2, n_hiddenlayers=2)
    vae.train(x_train, x_test, epochs=50, batch_size=100, early_stopping=True)
    vae.plot_embedding(x_test, y_test)
    samples = vae.sample_z(n_samples)


if model == "gan":
    if dataset == "mnist":
        x_train = x_train.reshape(-1, 28, 28, 1).astype(np.float32)
        mnist_dcgan = DCGAN(28, 28, 1, latent_dim=100)
        mnist_dcgan.train(x_train, train_steps=100, batch_size=256, save_interval=500)
        mnist_dcgan.show_samples(fake=True)
        mnist_dcgan.show_samples(fake=False, save2file=True)
        samples = mnist_dcgan.sample_G(n=n_samples)

    if dataset == "adult":
        gan = SimpleGAN((x_train.shape[1],), latent_dim=20)
        gan.train(x_train, train_steps=10000, batch_size=100)
        samples = gan.sample_G(n=n_samples)


if model == "rbm":
    rbm = BernoulliRBM(batch_size=100, verbose=1, n_iter=20)
    rbm.fit(x_train)

    v = np.random.randint(2, size=(n_samples, x_train.shape[1]))
    for i in range(2):
        v = rbm.gibbs(v)
    samples = v


'''
if model == "dbn":
    pretrain = False
    #trX, trY, vlX, vlY, teX, teY = datasets.load_mnist_dataset(mode='supervised')
    dbn = DeepBeliefNetwork(do_pretrain=pretrain, rbm_layers=[500, 10],
                        finetune_learning_rate=0.2, finetune_num_epochs=20)
    if pretrain:
        dbn.pretrain(x_train)
    dbn.fit(x_train, to_one_hot(y_train))
    print('Test set accuracy: {}'.format(dbn.score(x_test, to_one_hot(y_test))))
'''

report(X, y, samples, vec, model, dataset, binary_encoding, reoder_categories)


if dataset == "mnist":
    #plot_mnist_samples(vae)
    plt.figure(figsize=(10, 10))
    for x in samples:
        plt.imshow(x.reshape(28, 28), interpolation="none")
        plt.show()
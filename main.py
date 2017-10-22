from preprocessing.datasets import adult_dataset, mnist_dataset, binary_dataset, boston_houses_dataset
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import BernoulliRBM

from evaluation import *
from models.DCGAN import DCGAN
from models.SimpleGAN import SimpleGAN
from models.VAE import VAE
from preprocessing.Vectorizer import Vectorizer
import seaborn as sns


def main(dataset, model, binary_encoding, reorder_categories):
    print("> Preprocessing")
    vec = Vectorizer(binary=binary_encoding, feature_range=(-1, 1))

    if dataset == "adult":
        X, y = adult_dataset(drop_y=False)
        X_t = vec.fit_transform(X, True, True, reorder_categories)

    elif dataset == "mnist":
        X, y = mnist_dataset()
        X_t = vec.fit_transform(X, True, True, reorder_categories)

    elif dataset == "binary":
        X, y = binary_dataset()
        X_t = vec.fit_transform(X, True, False, reorder_categories)

    elif dataset == "boston":
        X, y = boston_houses_dataset()
        X_t = vec.fit_transform(X, True, True, reorder_categories)

    x_train, x_test, y_train, y_test = split_data(X_t, y)
    n_samples = len(X)

    print("> Training")
    if model == "gmm":
        modelparams = 12
        gmm = GaussianMixture(n_components=modelparams)
        gmm.fit(x_train)
        samples = gmm.sample(n_samples)[0]

    elif model == "vae":
        modelparams = [30, 30, 2]
        vae = VAE(intermediate_dim=modelparams[0], latent_dim=modelparams[-1], n_hiddenlayers=len(modelparams) - 1)
        vae.train(x_train, x_test, epochs=30, batch_size=128, early_stopping=True)
        #vae.plot_embedding(x_test, y_test)
        samples = vae.sample_z(n_samples)

    elif model == "gan":
        modelparams=[5]
        if dataset == "mnist":
            x_train = x_train.reshape(-1, 28, 28, 1).astype(np.float32)
            mnist_dcgan = DCGAN(28, 28, 1, latent_dim=modelparams[0])
            mnist_dcgan.train(x_train, train_steps=100, batch_size=128, save_interval=500)
            mnist_dcgan.show_samples(fake=True)
            mnist_dcgan.show_samples(fake=False, save2file=True)
            samples = mnist_dcgan.sample_G(n=n_samples)

        if dataset == "adult":
            gan = SimpleGAN((x_train.shape[1],), latent_dim=modelparams[0])
            gan.train(x_train, train_steps=10000, batch_size=128)
            samples = gan.sample_G(n=n_samples)

    elif model == "rbm":
        modelparams = [128]
        rbm = BernoulliRBM(n_components=modelparams[0], batch_size=32, verbose=1, n_iter=100)
        rbm.fit(x_train)

        v = np.random.randint(2, size=(n_samples, x_train.shape[1]))
        for i in range(100):
            v = rbm.gibbs(v)
        samples = v

    report(X, y, samples, vec, model, modelparams, dataset, binary_encoding, reorder_categories)


def test_all(trials=3):
    for m in ["gmm", "vae", "rbm", "gan"]:
        for b in range(2):
            for r in range(2):
                for _ in range(trials):
                    main(dataset, m, b, r)


dataset = "adult"
model = "gmm"
binary_encoding = True
reorder_categories = True
#main(dataset, model, binary_encoding, reorder_categories)

test_all()

'''  

def to_one_hot(y):
    if len(np.shape(y)) == 1:
        return np.eye(np.max(y)+1)[y]
    else:
        return y
        
        
if model == "dbn":
  pretrain = False
  #trX, trY, vlX, vlY, teX, teY = datasets.load_mnist_dataset(mode='supervised')
  dbn = DeepBeliefNetwork(do_pretrain=pretrain, rbm_layers=[500, 10],
                      finetune_learning_rate=0.2, finetune_num_epochs=20)
  if pretrain:
      dbn.pretrain(x_train)
  dbn.fit(x_train, to_one_hot(y_train))
  print('Test set accuracy: {}'.format(dbn.score(x_test, to_one_hot(y_test))))


  

if dataset == "mnist":
  #plot_mnist_samples(vae)
  plt.figure(figsize=(10, 10))
  for x in samples:
      plt.imshow(x.reshape(28, 28), interpolation="none")
      plt.show()
'''
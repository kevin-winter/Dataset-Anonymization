from preprocessing.datasets import adult_dataset, mnist_dataset, binary_dataset, boston_houses_dataset
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import BernoulliRBM
from sklearn.naive_bayes import GaussianNB, BernoulliNB

from evaluation import *
from models.DCGAN import DCGAN
from models.SimpleGAN import SimpleGAN
from models.VAE import VAE
from preprocessing.Vectorizer import Vectorizer
import seaborn as sns


def main(dataset, model, binary_encoding, reorder_categories):
    print("> Preprocessing")
    vec = Vectorizer(binary=binary_encoding, feature_range=[0, 1])

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
    modelparams = []

    print("> Training")
    if model == "gmm":
        modelparams = 12
        gmm = GaussianMixture(n_components=modelparams)
        gmm.fit(x_train)
        samples = gmm.sample(n_samples)[0]

    elif model == "vae":
        modelparams = [30, 30, 2]
        vae = VAE(intermediate_dim=modelparams[0], latent_dim=modelparams[-1], n_hiddenlayers=len(modelparams) - 1)
        vae.train(x_train, x_test, epochs=50, batch_size=64, early_stopping=True)
        vae.plot_embedding(x_test, y_test)

        if not os.path.exists("./results/figures"):
            os.makedirs("./results/figures")
        plt.savefig("./results/figures/latentspace_{}_{}_{}_{}".format(
            model, dataset, "binary" if binary_encoding else "cont", "reordered" if reorder_categories else "regular"))

        samples = vae.sample_z(n_samples)

    elif model == "gan":
        modelparams=[2]
        if dataset == "mnist":
            x_train = x_train.reshape(-1, 28, 28, 1).astype(np.float32)
            mnist_dcgan = DCGAN(28, 28, 1, latent_dim=modelparams[0])
            mnist_dcgan.train(x_train, train_steps=100, batch_size=128, save_interval=500)
            mnist_dcgan.show_samples(fake=True)
            mnist_dcgan.show_samples(fake=False, save2file=True)
            samples = mnist_dcgan.sample_G(n=n_samples)

            if not os.path.exists("./results/figures"):
                os.makedirs("./results/figures")
            plt.savefig("./results/figures/latentspace_{}_{}_{}_{}".format(
                model, dataset, "binary" if binary_encoding else "cont", "reordered" if reorder_categories else "regular"))

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

    elif model == "gnb":
        clf = GaussianNB()
        clf.fit(x_train, y_train)
        #sample
        c = np.random.choice(len(clf.classes_), p=clf.class_prior_)
        samples = []
        for i in range(n_samples):
            samples.append(np.random.multivariate_normal(clf.theta_[c], np.diag(clf.sigma_[c])))

    elif model == "bnb":
        clf = BernoulliNB(binarize=0.5)
        clf.fit(x_train, y_train)
        #sample
        pc = np.exp(clf.intercept_) if len(clf.classes_) > 2 else np.concatenate((np.exp(clf.intercept_), 1-np.exp(clf.intercept_)))
        c = np.random.choice(len(clf.classes_), p=pc)
        pv = np.exp(clf.coef_[c]) if len(clf.classes_) > 2 else np.vstack((np.exp(clf.coef_), 1-np.exp(clf.coef_)))[c]
        samples = []
        for i in range(n_samples):
            samples.append(np.random.binomial(1, p=pv))



    report(X, y, samples, vec, model, modelparams, dataset, binary_encoding, reorder_categories)


def test_all(trials=3):
    for m in ["gmm", "vae", "rbm", "gan"]:
        for b in range(2):
            for r in range(2):
                for _ in range(trials):
                    main(dataset, m, b, r)


dataset = "adult"
model = "gnb"
binary_encoding = False
reorder_categories = True
main(dataset, model, binary_encoding, reorder_categories)

#test_all()

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
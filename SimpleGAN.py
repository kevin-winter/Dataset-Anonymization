import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape, Conv2D, \
    Conv2DTranspose, UpSampling2D, LeakyReLU, Dropout, BatchNormalization

from GAN import GAN


class SimpleGAN(GAN):
    def __init__(self, input_dim, latent_dim):
        GAN.__init__(self, input_dim, latent_dim)

        self.intermediate_dim = (self.latent_dim + np.prod(self.input_dim)) // 2
        self.dropout = 0.4

        self.discriminator_model()
        self.adversarial_model()
        self.generator()

    def discriminator(self):
        if self.D:
            return self.D
        self.D = Sequential()
        dropout = 0.4

        self.D.add(Dense(self.intermediate_dim, input_shape=self.input_dim))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Dense(self.intermediate_dim))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(self.dropout))

        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        self.D.summary()
        return self.D

    def generator(self):
        if self.G:
            return self.G

        self.G = Sequential()

        self.G.add(Dense(self.intermediate_dim, input_dim=self.latent_dim))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Dropout(self.dropout))

        self.G.add(Dense(self.intermediate_dim))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Dropout(self.dropout))

        self.G.add(Dense(self.input_dim[0]))
        self.G.add(Activation('sigmoid'))
        self.G.summary()
        return self.G

    def show_samples(self, save2file=False, fake=True, n=16, noise=None, step=0):
        if fake:
            samples = self.sample_G(noise, n)
        else:
            i = np.random.randint(0, self.x_train.shape[0], n)
            samples = self.x_train[i, :, :, :]
        """
        plt.figure(figsize=(10, 10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, 0]
            image = np.reshape(image, [self.input_dim[0], self.input_dim[1]])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            filename = 'SimpleGAN_samples{}.png'.format("_{}".format(step) if step != 0 else "")
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()
        """
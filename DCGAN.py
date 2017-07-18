import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape, Conv2D, \
    Conv2DTranspose, UpSampling2D, LeakyReLU, Dropout, BatchNormalization

from GAN import GAN


class DCGAN(GAN):
    def __init__(self, img_rows=28, img_cols=28, channel=1, latent_dim=100):
        GAN.__init__(self, (img_rows, img_cols, channel), latent_dim)

        self.discriminator_model()
        self.adversarial_model()
        self.generator()

    def discriminator(self):
        if self.D:
            return self.D
        self.D = Sequential()
        depth = 64
        dropout = 0.4

        self.D.add(Conv2D(depth*1, 5, strides=2, input_shape=self.input_dim, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*2, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*8, 5, strides=1, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        self.D.summary()
        return self.D

    def generator(self):
        if self.G:
            return self.G
        self.G = Sequential()
        self.latent_dim = 100
        dropout = 0.4
        depth = 64+64+64+64
        dim = 7

        self.G.add(Dense(dim*dim*depth, input_dim=self.latent_dim))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((dim, dim, depth)))
        self.G.add(Dropout(dropout))

        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(Conv2DTranspose(1, 5, padding='same'))
        self.G.add(Activation('sigmoid'))
        self.G.summary()
        return self.G

    def show_samples(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        if fake:

            images = self.G.predict(noise)
        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        plt.figure(figsize=(10, 10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, 0]
            image = np.reshape(image, [self.input_dim[0], self.input_dim[1]])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            filename = 'DCGAN_samples{}.png'.format("_{}".format(step) if step != 0 else "")
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()


if __name__ == '__main__':
    x_train = input_data.read_data_sets("MNIST_data", one_hot=True)\
        .train.images.reshape(-1, 28, 28, 1).astype(np.float32)
    mnist_dcgan = DCGAN(28, 28, 1)
    mnist_dcgan.train(x_train, train_steps=2000, batch_size=256, save_interval=100)
    mnist_dcgan.show_samples(fake=True)
    mnist_dcgan.show_samples(fake=False, save2file=True)
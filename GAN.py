import numpy as np
from keras.models import Sequential
from keras.optimizers import RMSprop


class GAN:
    def __init__(self, input_dim, latent_dim):
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    def discriminator(self):
        raise NotImplementedError("Please define a discriminator model")

    def generator(self):
        raise NotImplementedError("Please define a generator model")

    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self.AM

    def train(self, x_train, train_steps=2000, batch_size=256, save_interval=0):
        self.x_train = x_train
        noise_input = None
        if save_interval > 0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, self.latent_dim])
        for i in range(train_steps):
            train = self.x_train[np.random.randint(0, self.x_train.shape[0], size=batch_size)]
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, self.latent_dim])
            fake = self.G.predict(noise)
            x = np.concatenate((train, fake))
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.DM.train_on_batch(x, y)

            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, self.latent_dim])
            a_loss = self.AM.train_on_batch(noise, y)

            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
            if save_interval > 0:
                if (i+1) % save_interval == 0:
                    self.show_samples(save2file=True, n=noise_input.shape[0], noise=noise_input, step=(i+1))

    def sample_G(self, noise=None, n=16):
        if noise is None:
            noise = np.random.uniform(-1.0, 1.0, size=[n, self.latent_dim])
        return self.G.predict(noise)

    def show_samples(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        raise NotImplementedError("Please define a generator model")

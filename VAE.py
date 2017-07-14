import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.callbacks import EarlyStopping


class VAE():
    class CustomVariationalLayer(Layer):
        def __init__(self, vae, **kwargs):
            self.vae = vae
            self.is_placeholder = True
            super(vae.CustomVariationalLayer, self).__init__(**kwargs)

        def vae_loss(self, x, x_decoded_mean):
            xent_loss = self.vae.original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.sum(1 + self.vae.z_log_var - K.square(self.vae.z_mean) - K.exp(self.vae.z_log_var), axis=-1)
            return K.mean(xent_loss + kl_loss)

        def call(self, inputs, **kwargs):
            self.add_loss(self.vae_loss(inputs[0], inputs[1]), inputs=inputs)
            return inputs[0]

    def __init__(self, n_hiddenlayers, intermediate_dim=256, latent_dim=2, epsilon_std=1.0):
        self.n_hiddenlayers = n_hiddenlayers
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.epsilon_std = epsilon_std

    def init_model(self, original_dim, batch_size):
        self.batch_size = batch_size
        self.original_dim = original_dim

        # Create En- and Decoder for training
        x = Input(batch_shape=(batch_size, original_dim))
        for i in range(self.n_hiddenlayers):
            if i == 0: h = x
            h = Dense(self.intermediate_dim, activation='relu')(h)
        self.z_mean = Dense(self.latent_dim)(h)
        self.z_log_var = Dense(self.latent_dim)(h)
        z = Lambda(self._sampling)([self.z_mean, self.z_log_var])

        decoder_layers = []
        decoder_layers.append(Dense(self.intermediate_dim, activation='relu', input_dim=(self.latent_dim,)))
        for i in range(self.n_hiddenlayers-1):
            decoder_layers.append(Dense(self.intermediate_dim, activation='relu'))
        decoder_mean = Dense(original_dim, activation='sigmoid')

        for i in range(self.n_hiddenlayers):
            if i == 0: hd = z
            hd = decoder_layers[i](hd)
        x_decoded_mean = decoder_mean(hd)
        y = self.CustomVariationalLayer(self)([x, x_decoded_mean])

        vae = Model(x, y)
        vae.compile(optimizer='rmsprop', loss=None)

        # Create Encoder
        encoder = Model(x, self.z_mean)

        # Create Decoder
        decoder_input = Input(shape=(self.latent_dim,))
        for i in range(self.n_hiddenlayers):
            if i == 0: _hd = decoder_input
            _hd = decoder_layers[i](_hd)
        _x_decoded_mean = decoder_mean(_hd)
        generator = Model(decoder_input, _x_decoded_mean)

        self._vae = vae
        self._encoder = encoder
        self._generator = generator

    def _sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(self.batch_size, self.latent_dim), mean=0., stddev=self.epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def train(self, x_train, x_test, epochs=50, batch_size=100, early_stopping=True):
        callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')] if early_stopping else []
        self.init_model(x_train.shape[1], batch_size=batch_size)
        x_train = x_train[:len(x_train)//batch_size*batch_size]
        x_test = x_test[:len(x_test)//batch_size*batch_size]
        self._vae.fit(x_train, shuffle=True, epochs=epochs, batch_size=self.batch_size,
                      validation_data=(x_test, x_test), callbacks=callbacks)

    def sample_z(self, n):
        X = np.random.normal(size=(n, self.latent_dim))
        y = self._generator.predict(X, batch_size=n)
        return y

    def plot_embedding(self, x_test, y_test):
        x_test = x_test[:len(x_test)//self.batch_size*self.batch_size]
        y_test = y_test[:len(x_test)//self.batch_size*self.batch_size]
        x_test_encoded = self._encoder.predict(x_test, batch_size=self.batch_size)
        plt.figure(figsize=(6, 6))
        plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
        plt.colorbar()
        plt.show()

    def plot_samples(self):
        n = 15
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))
        grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.array([[xi, yi]])
                x_decoded = self._generator.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='Greys_r')
        plt.show()

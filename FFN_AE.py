from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from typing import Any
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from model_base import AutoencoderBaseModel


class FFNModelMixin:
    @staticmethod
    def ffn_autoencoder(x_train,
                        x_train_target,
                        x_test,
                        x_test_target,
                        epochs: int = 10,
                        batch_size: int = 128,
                        optimizer: str = "adam"):
        input_layer = Input(shape=(784,))
        encoded = Dense(256, activation='relu')(input_layer)
        encoded = Dense(128, activation='relu')(encoded)
        encoded = Dense(64, activation='relu')(encoded)
        encoded = Dense(32, activation='relu')(encoded)
        decoded = Dense(64, activation='relu')(encoded)
        decoded = Dense(128, activation='relu')(decoded)
        decoded = Dense(256, activation='relu')(decoded)
        output_layer = Dense(784, activation='sigmoid')(decoded)

        autoencoder = Model(input_layer, output_layer)
        autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')
        autoencoder.summary()
        history = autoencoder.fit(x_train, x_train_target,
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  validation_data=(x_test, x_test_target),
                                  verbose=1)
        return autoencoder, history


class FFNAutoencoder(AutoencoderBaseModel, FFNModelMixin):
    def __init__(self):
        super().__init__(model_type="FFN")

    @staticmethod
    def load_and_prepare_mnist(train_size: int = 60000,
                               test_size: int = 10000) -> Any:
        (x_train, _), (x_test, _) = mnist.load_data()
        x_train = x_train[0:train_size]
        x_test = x_test[0:test_size]
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        return x_train, x_test

    @staticmethod
    def inject_noise_mnist(x_train: Any,
                           x_test: Any,
                           noise_sd: float = 0.5,
                           scale: bool = True) -> Any:
        x_train_noisy = x_train + np.random.normal(0, noise_sd, size=x_train.shape)
        x_test_noisy = x_test + np.random.normal(0, noise_sd, size=x_test.shape)
        if scale:
            scaler = MinMaxScaler()
            scaler.fit(x_train_noisy)
            x_train_noisy = scaler.transform(x_train_noisy)
            x_test_noisy = scaler.transform(x_test_noisy)
        return x_train_noisy, x_test_noisy

    def fit(self,
            x_train,
            x_train_target: Any,
            x_test: Any,
            x_test_target: Any,
            epochs: int = 20,
            batch_size: int = 128,
            optimizer: str = "adam"):
        fitted_model, history = self.ffn_autoencoder(x_train, x_train_target, x_test, x_test_target,
                                                     epochs=epochs,
                                                     batch_size=batch_size,
                                                     optimizer=optimizer)
        self.model = fitted_model
        self.train_history = history

    def predict(self, x_input):
        return self.model.predict(x_input)


if __name__ == "__main__":
    train = True
    # train = False
    noise_sd = 0.5
    if train:
        model = FFNAutoencoder()
        x_train, x_test = model.load_and_prepare_mnist(train_size=5000, test_size=500)
        x_train_noisy, x_test_noisy = model.inject_noise_mnist(x_train,
                                                               x_test,
                                                               noise_sd=noise_sd,
                                                               scale=True)
        model.fit(x_train_noisy, x_train, x_test_noisy, x_test,
                  epochs=3,
                  batch_size=256,
                  optimizer="adam")
        model.save("FFN_autoencoder.h5")
        preds = model.predict(x_test_noisy)
        model.plot_mnist_preds(x_test_noisy, preds)
        model.plot_learning()
    else:
        model = FFNAutoencoder()
        model.load("FFN_autoencoder.h5")
        x_train, x_test = model.load_and_prepare_mnist()
        x_train_noisy, x_test_noisy = model.inject_noise_mnist(x_train, x_test, noise_sd=noise_sd)
        random_rows = model.sample_unique_ints(10, 0, len(x_test))
        model.plot_mnist_preds(x_test_noisy[random_rows], model.predict(x_test_noisy[random_rows]))

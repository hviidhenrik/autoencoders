from model_base import AutoencoderBaseModel
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
from typing import Any
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class CNNModelMixin:
    @staticmethod
    def cnn_autoencoder(x_train,
                        x_train_target,
                        x_test,
                        x_test_target,
                        epochs: int = 10,
                        batch_size: int = 128,
                        optimizer: str = "adam"):
        input_layer = Input(shape=(28, 28, 1))
        encoding = Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
        encoding = MaxPooling2D((2, 2), padding='same')(encoding)
        encoding = Conv2D(8, (3, 3), activation='relu', padding='same')(encoding)
        encoding = MaxPooling2D((2, 2), padding='same')(encoding)
        encoding = Conv2D(8, (3, 3), activation='relu', padding='same')(encoding)
        encoding = MaxPooling2D((2, 2), padding='same')(encoding)

        decoding = Conv2D(8, (3, 3), activation='relu', padding='same')(encoding)
        decoding = UpSampling2D((2, 2))(decoding)
        decoding = Conv2D(8, (3, 3), activation='relu', padding='same')(decoding)
        decoding = UpSampling2D((2, 2))(decoding)
        decoding = Conv2D(16, (3, 3), activation='relu')(decoding)
        decoding = UpSampling2D((2, 2))(decoding)
        output_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoding)

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


class CNNAutoencoder(AutoencoderBaseModel, CNNModelMixin):
    def __init__(self):
        super().__init__(model_type="CNN")

    def fit(self,
            x_train,
            x_train_target: Any,
            x_test: Any,
            x_test_target: Any,
            epochs: int = 20,
            batch_size: int = 128,
            optimizer: str = "adam"):
        fitted_model, history = self.cnn_autoencoder(x_train, x_train_target, x_test, x_test_target,
                                                     epochs=epochs,
                                                     batch_size=batch_size,
                                                     optimizer=optimizer)
        self.model = fitted_model
        self.train_history = history

    def inject_noise_mnist(self,
                           x_train: Any,
                           x_test: Any,
                           noise_sd: float = 0.5,
                           scale: bool = True) -> Any:
        x_train_noisy, x_test_noisy = super().inject_noise_mnist(x_train, x_test, noise_sd, scale)
        if scale:
            x_train_noisy = x_train_noisy / x_train_noisy.max()
            x_test_noisy = x_test_noisy / x_test_noisy.max()
        return x_train_noisy, x_test_noisy

    def predict(self, x_input):
        return self.model.predict(x_input)

if __name__ == "__main__":
    train = True
    # train = False
    noise_sd = 0.5
    if train:
        mymodel = CNNAutoencoder()
        x_train, x_test = mymodel.load_and_prepare_mnist()
        x_train_noisy, x_test_noisy = mymodel.inject_noise_mnist(x_train,
                                                                 x_test,
                                                                 noise_sd=noise_sd,
                                                                 scale=True)
        mymodel.fit(x_train, x_train, x_test, x_test,
                    epochs=20,
                    batch_size=256,
                    optimizer="adam")
        mymodel.save("CNN_autoencoder.h5")
        preds = mymodel.predict(x_test_noisy)
        mymodel.plot_mnist_preds(x_test_noisy, preds)
        mymodel.plot_learning()
    else:
        mymodel = CNNAutoencoder()
        mymodel.load("CNN_autoencoder.h5")
        x_train, x_test = mymodel.load_and_prepare_mnist()
        x_train_noisy, x_test_noisy = mymodel.inject_noise_mnist(x_train, x_test, noise_sd=noise_sd)
        random_rows = mymodel.sample_unique_ints(10, 0, len(x_test))
        mymodel.plot_mnist_preds(x_test_noisy[random_rows], mymodel.predict(x_test_noisy[random_rows]))
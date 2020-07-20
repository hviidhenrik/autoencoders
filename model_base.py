import silence_tensorflow.auto
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.datasets import mnist
from typing import Any

from sklearn.preprocessing import MinMaxScaler


class PlotsMixin:
    def plot_learning(self):
        history = self.train_history
        plt.plot(history.history['loss'], label="train")
        plt.plot(history.history['val_loss'], label="validation")
        plt.title("Learning curve")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(loc="upper right")
        plt.show()

    def plot_mnist_preds(self, x_input, x_predictions, N_digits=10):
        plt.figure(figsize=(20, 4))
        for i in range(N_digits):
            # display original
            ax = plt.subplot(2, N_digits, i + 1)
            plt.imshow(x_input[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, N_digits, i + 1 + N_digits)
            plt.imshow(x_predictions[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()


class GeneralMixin:
    def sample_unique_ints(self, N_ints: int = 10, min: int = 0, max: int = 100):
        integers = []
        while len(integers) < N_ints:
            i = np.random.randint(min, max)
            if i not in integers: integers.append(i)
        return integers


class AutoencoderBaseModel(ABC, PlotsMixin, GeneralMixin):
    def __init__(self, model_type):
        self.model_type = model_type
        self.train_history = None
        self.model = None
        super().__init__()

    @staticmethod
    def load_and_prepare_mnist(train_size: int = 60000,
                               test_size: int = 10000) -> Any:
        (x_train, _), (x_test, _) = mnist.load_data()
        x_train = x_train[0:train_size]
        x_test = x_test[0:test_size]
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        return x_train, x_test

    @staticmethod
    def inject_noise_mnist(x_train: Any,
                           x_test: Any,
                           noise_sd: float = 0.5,
                           scale: bool = True) -> Any:
        x_train_noisy = x_train + np.random.normal(0, noise_sd, size=x_train.shape)
        x_test_noisy = x_test + np.random.normal(0, noise_sd, size=x_test.shape)
        return x_train_noisy, x_test_noisy

    # @abstractmethod
    # def fit(self):
    #     pass

    # @abstractmethod
    # def predict(self, x_input):
    #     pass

    def save(self, filename):
        self.model.save(filename)
        print("Saved model to disk as " + filename)

    def load(self, filename):
        self.model = load_model(filename)
        print("Model loaded and ready")

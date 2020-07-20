from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model



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
        super().__init__()

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

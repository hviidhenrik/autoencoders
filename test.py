from keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = x_train[0:1000]
x_test = x_test[0:100]

input_dim = (x_train.shape[1],x_train.shape[2], 1)

# define model
# input_layer = Input(shape=input_dim)
# encoder = Conv2D(1, (3,3),activation="relu", padding='same')(input_layer)
#
# encoder = MaxPooling2D((2,2), padding='same')(encoder)
# output_layer = Conv2D(1, (3,3), activation="sigmoid", padding='same')(encoder)

input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (7, 7, 32)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
autoencoder = Model(input_img, decoded)


# autoencoder = Model(input_layer, output_layer)
autoencoder.summary()
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=128,
                validation_data=(x_test, x_test),
                verbose=1)




from model import DCGAN


from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import sys
import numpy as np


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Load the dataset
    (X_train, _), (_, _) = mnist.load_data()
    # Rescale -1 to 1
    X_train = X_train / 127.5 - 1.
    X_train = np.expand_dims(X_train, axis=3)
    print(X_train.shape[0])
    dcgan = DCGAN(imageShape=(X_train.shape[1],X_train.shape[2],X_train.shape[3]),latentShape=100)
    dcgan.fit(xTrain=X_train,epochs=4000, batch_size=32, save_interval=50)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

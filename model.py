from Layers.Discriminator import Discriminator
from Layers.Generators import Generator

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


class DCGAN():
    def __init__(self, imageShape, latentShape):
        self.imageShape = imageShape
        self.latentShape = latentShape
        self.generator = Generator(self.imageShape, self.latentShape)
        self.generator = self.generator.BuildModel()
        self.discriminator = Discriminator(self.imageShape)
        self.discriminator = self.discriminator.BuildModel()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=Adam(0.0002, 0.5),
                                   metrics=['accuracy'])

        # Build the generator
        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latentShape,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

    def fit(self, xTrain, epochs, batch_size=128, save_interval=50):
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, xTrain.shape[0], batch_size)
            imgs = xTrain[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latentShape))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
            r, c = 5, 5
            noise = np.random.normal(0, 1, (r * c, self.latentShape))
            gen_imgs = self.generator.predict(noise)

            # Rescale images 0 - 1
            gen_imgs = 0.5 * gen_imgs + 0.5
            fig, axs = plt.subplots(r, c)
            cnt = 0

            for i in range(r):
                for j in range(c):
                    axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                    axs[i, j].axis('off')
                    cnt += 1

            fig.savefig("images/mnist_%d.png" % epoch)
            plt.close

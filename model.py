from Layers.Discriminator import Discriminator
from Layers.Generators import Generator
import time
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
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display


# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)



def train_step(images, generator, discriminator,generator_model,discriminator_model):
    EPOCHS = 50
    noise_dim = 100
    num_examples_to_generate = 16
    BUFFER_SIZE = 60000
    BATCH_SIZE = 256
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator_model(noise, training=True)

        real_output = discriminator_model(images, training=True)
        fake_output = discriminator_model(generated_images, training=True)

        gen_loss = generator.generator_loss(fake_output)
        disc_loss = discriminator.discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator_model.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator_model.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_model.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator_model.trainable_variables))


class DCGAN():
    def __init__(self, latentShape):

        self.latentShape = latentShape


        self.num_examples_to_generate = 16
        self.BUFFER_SIZE = 60000
        self.EPOCHS = 50
        self.seed = tf.random.normal([self.num_examples_to_generate, latentShape])

        # Build the generator
        # The generator takes noise as input and generates imgs
        self.generator = Generator()
        self.generator_model = self.generator.BuildModel()

        noise = tf.random.normal([1, 100])
        generated_image = self.generator_model(noise, training=False)

        # For the combined model we will only train the generator

        # The discriminator takes generated images as input and determines validity
        self.discriminator = Discriminator()
        self.discriminator_model = self.discriminator.BuildModel()
        decision = self.discriminator_model(generated_image)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator


    def generate_and_save_images(self, model, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig("images/mnist_%d.png" % epoch)
        plt.close

    def train(self, dataset, epochs):
        for epoch in range(epochs):
            start = time.time()

            for image_batch in dataset:
                train_step(image_batch,self.generator,self.discriminator,self.generator_model,self.discriminator_model)

            # Produce images for the GIF as you go
            display.clear_output(wait=True)
            self.generate_and_save_images(self.generator_model,
                                          epoch + 1,
                                          self.seed)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        # Generate after the final epoch
        display.clear_output(wait=True)
        self.generate_and_save_images(self.generator_model,
                                      epochs,
                                      self.seed)

from Layers.Discriminator import Discriminator
from Layers.Generators import Generator
import tensorflow as tf
import matplotlib.pyplot as plt

import time

from IPython import display


def TrainStep(images, generator, discriminator, generatorModel, discriminatorModel):
    EPOCHS = 50
    noise_dim = 100
    num_examples_to_generate = 16
    BUFFER_SIZE = 60000
    BATCH_SIZE = 256
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generatorModel(noise, training=True)

        realOutput = discriminatorModel(images, training=True)
        fakeOutput = discriminatorModel(generated_images, training=True)

        genLoss = generator.GeneratorLoss(fakeOutput)
        discLoss = discriminator.DiscriminatorLoss(realOutput, fakeOutput)

    gradientsOfGenerator = gen_tape.gradient(genLoss, generatorModel.trainable_variables)
    gradientsOfDiscriminator = disc_tape.gradient(discLoss, discriminatorModel.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradientsOfGenerator, generatorModel.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradientsOfDiscriminator, discriminatorModel.trainable_variables))
    return [genLoss, discLoss]


class DCGAN():
    def __init__(self, latentShape):

        self.latentShape = latentShape

        self.num_examples_to_generate = 16
        self.BUFFER_SIZE = 60000
        self.EPOCHS = 50
        self.seed = tf.random.normal([self.num_examples_to_generate, latentShape])

        # Build the generator
        # The generator takes noise as input and generates images
        self.generator = Generator()
        self.generatorModel = self.generator.BuildModel()

        noise = tf.random.normal([1, 100])
        generated_image = self.generatorModel(noise, training=False)

        # For the combined model we will only train the generator

        # The discriminator takes generated images as input and determines validity
        self.discriminator = Discriminator()
        self.discriminatorModel = self.discriminator.BuildModel()
        decision = self.discriminatorModel(generated_image)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator

    def SaveImage(self, model, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode
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
                result = TrainStep(image_batch, self.generator, self.discriminator, self.generatorModel,
                                   self.discriminatorModel)

            display.clear_output(wait=True)
            self.SaveImage(self.generatorModel,
                           epoch + 1,
                           self.seed)

            print('Generate Loss : {} , Discriminator Loss {}'.format(result[0], result[1]))
            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        # Generate after the final epoch
        display.clear_output(wait=True)
        self.SaveImage(self.generatorModel,
                       epochs,
                       self.seed)

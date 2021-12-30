from model import DCGAN
import tensorflow as tf

if __name__ == '__main__':
    (TrainImage, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    TrainImage = TrainImage.reshape(TrainImage.shape[0], 28, 28, 1).astype('float32')
    TrainImage = (TrainImage - 127.5) / 127.5  # Normalize the images to [-1, 1]

    BUFFER_SIZE = 60000
    BATCH_SIZE = 256
    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(TrainImage).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    dcgan = DCGAN(latentShape=100)
    dcgan.train(train_dataset,75)



import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#import tensorflow_examples.models.pix2pix
import tensorflow_datasets as tfds

def loadImageTrain(datapoint):
    image = tf.image.resize(datapoint["image"], (128, 128))
    mask = tf.image.resize(datapoint["segmentation_mask"], (128, 128))
    
    if(tf.random.uniform() > 0.5):
        image = tf.image.flip_left_right(image)
        mask = tf.image.fli_lef_right(mask)
        
    return image, mask

def loadImageTest(datapoint):
    image = tf.image.resize(datapoint["image"], (128, 128))
    mask = tf.image.resize(datapoint["segmentation_mask"], (128, 128))
    
    image, mask = normaliseImage(image, mask)
    
    return image, mask

def normaliseImage(image, mask):
    image = tf.cast(image, tf.float32) / 255.0
    mask -= 1 # makes classes [0, 1, 2] instead of [1, 2, 3]
    
    return image, mask

def main():
    fname = "images.tar.gz"
    origin = "https://www.robots.ox.ac.uk/~vgg/data/pets/data\images.tar"
    dataset, info = tf.keras.utils.get_file(fname, origin, extract=True)

    #dataset, info = tfds.load("oxford_iiit_pet:3.*.*", with_info=True)
    TRAIN_LENGTH = info.splits['train'].num_examples
    BATCH_SIZE = 64
    BUFFER_SIZE = 1000
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
    
    train = dataset["train"].map(loadImageTrain, num_parrallel_calls=tf.data.experimental.AUTOTUNE)
    test = dataset["test"].map(loadImageTest)

if(__name__ == "__main__"):
    main()
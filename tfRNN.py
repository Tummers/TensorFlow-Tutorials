import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

def loadData():
    batch_size = 64
    seed = 42
    train_ds = tf.keras.preprocessing.text_dataset_from_directory("Imdb/aclImdb/train",
                                                                 batch_size=batch_size,
                                                                 validation_split=0.2,
                                                                 subset="training",
                                                                 seed=seed)
    val_ds = tf.keras.preprocessing.text_dataset_from_directory("Imdb/aclImdb/train",
                                                                 batch_size=batch_size,
                                                                 validation_split=0.2,
                                                                 subset="validation",
                                                                 seed=seed)
    
    test_ds = tf.keras.preprocessing.text_dataset_from_directory("Imdb/aclImdb/test",
                                                                 batch_size=batch_size)
    return train_ds, val_ds, test_ds

def altLoadData():
    dataset, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
    train, test = dataset["train"], dataset["test"]
    
    return train, test

def main():
    train, test = loadData()
    
    buffer_size = tf.data.experimental.AUTOTUNE
    batch_size = 64
    
    train = train.shuffle(buffer_size).batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)
    test = test.shuffle(buffer_size).batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)
    encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=1000)
    encoder.adapt(train.map(lambda text, label: text))

if(__name__ == "__main__"):
    main()
    

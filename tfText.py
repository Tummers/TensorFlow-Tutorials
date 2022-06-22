import io
import os
import re
import shutil
import string
import tensorflow as tf

from datetime import datetime
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

def getData():
    
    data_dir = os.path.abspath("C:\\Users\\tomjs\\Documents\\Python Scripts\\TensorFlow\\Imdb")
    data_dir = os.path.join(data_dir, "aclImdb")
    
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    
    batch_size = 1024
    seed = 123
    
    train_ds = tf.keras.preprocessing.text_dataset_from_directory(train_dir,
                                                                 batch_size=batch_size,
                                                                 validation_split=0.2,
                                                                 subset="training",
                                                                 seed=seed)
    val_ds = tf.keras.preprocessing.text_dataset_from_directory(train_dir,
                                                               batch_size=batch_size,
                                                               validation_split=0.2,
                                                               subset="validation",
                                                               seed=seed)
    
    test_ds = tf.keras.preprocessing.text_dataset_from_directory(test_dir)
    
    return train_ds, val_ds, test_ds

def customStandardisation(input_data):
    """
    standardises input text. makes lower case and removes punctuation
    """
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    final_string = tf.strings.regex_replace(stripped_html,
                                            "[%s]" % re.escape(string.punctuation), "")
    return final_string

def buildModel(text_only_ds):
    
    # converting the words into numerical values
    vocab_size = 10000
    sequence_length = 100
    vectorisation_layer = TextVectorization(standardize=customStandardisation,
                                            max_tokens=vocab_size,
                                            output_mode="int",
                                            output_sequence_length=sequence_length)
    
    # uses text_only_ds to build vocab values
    vectorisation_layer.adapt(text_only_ds)
    
    # embedding values into vectors,
    # vocab of 1000 words, each word becomes a 5d vector
    # embedding layers are initialised randomly and trained like the others
    embedding_layer = tf.keras.layers.Embedding(vocab_size, 5)
    
    model = Sequential([vectorisation_layer,
                        embedding_layer,
                        GlobalAveragePooling1D(),
                        Dense(16, activation="relu"),
                        Dense(1)])
    
    return model
    
def main():
    
    train, val, test = getData()
    
    # prefetching data to prevent i/o slowdown
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    
    train = train.cache().prefetch(buffer_size=AUTOTUNE)
    val = val.cache().prefetch(buffer_size=AUTOTUNE)
    
    # making text_only data
    text_ds = train.map(lambda x, y: x)
    
    # build and compile model
    model = buildModel(text_ds)
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
    
    model.compile(optimizer="adam",
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    
    model.fit(train,
              validation_data=val,
              epochs=15,
              callbacks=[tensorboard_callback])

if(__name__ == "__main__"):
    main()
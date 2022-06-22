import tensorflow as tf
import string
import re

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import layers
from tensorflow.keras import losses

def load_training():
    batch_size = 32
    seed = 42
    raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory("Imdb/aclImdb/train",
                                                                      batch_size=batch_size,
                                                                      validation_split=0.2,
                                                                      subset="training",
                                                                      seed=seed)
    
    raw_validation_ds = tf.keras.preprocessing.text_dataset_from_directory("Imdb/aclImdb/train",
                                                                      batch_size=batch_size,
                                                                      validation_split=0.2,
                                                                      subset="validation",
                                                                      seed=seed)
    
    raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory("Imdb/aclImdb/test",
                                                                    batch_size=batch_size)
    
    return raw_train_ds, raw_validation_ds, raw_test_ds

def standardisation(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    
    standardised_string = tf.strings.regex_replace(stripped_html, "[%s]" % re.escape(string.punctuation), " ")
    
    return standardised_string

def vectorise_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorisation_layer(text), label
    
if(__name__ == "__main__"):
    
    raw_training_ds, raw_val_ds, raw_test_ds = load_training()
    
    # text pre-processing
    max_features = 10000
    sequence_length = 250
    vectorisation_layer = TextVectorization(standardize=standardisation,
                                            max_tokens=max_features,
                                            output_mode="int",
                                            output_sequence_length=sequence_length)
    
    train_text = raw_training_ds.map(lambda x, y: x)
    vectorisation_layer.adapt(train_text)
    
    training_ds = raw_training_ds.map(vectorise_text)
    val_ds = raw_val_ds.map(vectorise_text)
    test_ds = raw_test_ds.map(vectorise_text)
    
    # cache-ing data we'll need to avoid slow down during training 
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    
    training_ds = training_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    embedding_dim = 16
    
    # building the model
    model = tf.keras.Sequential([layers.Embedding(max_features + 1, embedding_dim),
                                 layers.Dropout(0.2),
                                 layers.GlobalAveragePooling1D(),
                                 layers.Dropout(0.2),
                                 layers.Dense(1)])
    
    #model.summary()

    model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
                  optimizer="adam",
                  metrics=tf.metrics.BinaryAccuracy(threshold=0.0))
    
    # training
    epochs = 10
    history = model.fit(training_ds,
                        validation_data=val_ds,
                        epochs=epochs)
    
    # putting the text preprocessor into the model
    export_model = tf.keras.Sequential([vectorisation_layer, model, layers.Activation("sigmoid")])
    
    export_model.compile(loss=losses.BinaryCrossentropy(from_logits=False), 
                         optimizer="adam", 
                         metrics=["accuracy"])

    loss, accuracy = export_model.evaluate(raw_test_ds)
    print(accuracy)
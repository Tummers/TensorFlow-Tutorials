import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import tempfile

def inputFunction():
    """
    estimator input function, must return tf.data.Dataset
    """
    split = tfds.Split.TRAIN
    
    dataset = tfds.load("iris", split=split, as_supervised=True)
    dataset = dataset.map(lambda features, labels: ({'dense_input':features}, labels))
    dataset = dataset.batch(32).repeat()
    
    return dataset
    
def main():
    """
    standard method for building a keras model
    """
    keras_model = tf.keras.models.Sequential([tf.keras.layers.Dense(16, activation="relu", input_shape=(4, )),
                                              tf.keras.layers.Dropout(0.2),
                                              tf.keras.layers.Dense(3)])
    
    keras_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer="adam", metrics=["accuracy"])
    
    """
    make a temp directory for making estimator
    """
    model_dir = tempfile.mkdtemp()
    """
    single function call to produce an estimator
    """
    keras_estimator = tf.keras.estimator.model_to_estimator(keras_model=keras_model, model_dir=model_dir)
    
    keras_estimator.train(input_fn=inputFunction, steps=500)
    result = keras_estimator.evaluate(input_fn=inputFunction, steps=10)
    
    print(result)

if(__name__ == "__main__"):
    main()
import tensorflow as tf
import numpy as np
import io
import os
import re
import string
import tqdm

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import layers
from tensorflow.keras import Model, Sequential

class Word2Vec(Model):
    def __init__(self, vocab_size, embedding_dim, num_ns):
        super(Word2Vec, self).__init__()
        self.target_embedding = tf.keras.layers.Embedding(vocab_size,
                                                          embedding_dim,
                                                          input_length=1,
                                                          name="w2v_embedding")
        self.context_embedding = tf.keras.layers.Embedding(vocab_size,
                                                          embedding_dim,
                                                          input_length=num_ns + 1)
              
        self.dots = tf.keras.layers.Dot(axes=(3, 2))
        self.flatten = tf.keras.layers.Flatten()
        
    def call(self, pair):
        target, context = pair
        we = self.target_embedding(target)
        ce = self.context_embedding(context)
        dots = self.dots([ce, we])
        return self.flatten(dots)
    
def customLoss(x_logit, y_true):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)

def getData():
    file_path = tf.keras.utils.get_file("shakespeare.txt", "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")
    
    # drops empty lines
    text_dataset = tf.data.TextLineDataset(file_path).filter(lambda x: tf.cast(tf.strings.length(x), bool))
    
    return text_dataset

def standardisation(input_data):
    lowercase = tf.strings.lower(input_data)
    
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(string.punctuation), " ")
    
    
def skipGrams(strings, vocab_size, window_size, num_ns, SEED):
    """
    takes int encoded sequences of strings, returns dataset
    """
    targets, contexts, labels = [], [], []
    
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)
    
    for sentence in tqdm.tqdm(strings):
    
        # this generates positive cases of skip grams,
        # i.e. words which are neighbouring the target words
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(sentence,
                                                                           vocabulary_size=vocab_size,
                                                                           sampling_table=sampling_table,
                                                                           window_size=window_size,
                                                                           negative_samples=0)
       
        # this generates negative case skipgrams, 
        # words which aren't neighbours, but are in the vocabulary
        # number of neg samples num_ns should range from 5-20 for small datasets, 2-5 in large ones
        
        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(tf.constant([context_word], dtype="int64"), 1)
            
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(true_classes=context_class,
                                                                                         num_true=1,
                                                                                         num_sampled=num_ns,
                                                                                         unique=True,
                                                                                         range_max=vocab_size,
                                                                                         seed=SEED,
                                                                                         name="negative_sampling")
            
            # now we have positive and negative context words, we want
            # to put them in the same tensor as a single training example
            
            # add a dimension to negative candidates so we can concatenate
            negative_sampling_candidates = tf.expand_dims(negative_sampling_candidates, 1)
            
            context = tf.concat([context_class, negative_sampling_candidates], 0)
            
            # label the first context as positive, others as negative
            label = tf.constant([1] + [0] * num_ns, dtype="int64")
        
            targets.append(target_word)
            contexts.append(context)
            labels.append(label)
    
    return targets, contexts, labels
    

def main():
    SEED = 42
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    
    # downloading the data
    text_ds = getData()
    
    vocab_size = 4096
    sequence_length = 10

    # standardising and vectorising the data
    vectorise_layer = TextVectorization(standardize=standardisation,
                                        max_tokens=vocab_size,
                                        output_mode="int",
                                        output_sequence_length=sequence_length)


    vectorise_layer.adapt(text_ds.batch(1024))
    
    inverse_vocab = vectorise_layer.get_vocabulary()
    
    text_vector_ds = text_ds.batch(1024).prefetch(AUTOTUNE).map(vectorise_layer).unbatch()
    
    # breaking data into sentences / sequences
    
    sequences = list(text_vector_ds.as_numpy_iterator())
    num_ns = 4
    targets, contexts, labels = skipGrams(sequences, vocab_size, window_size=2, num_ns=num_ns, SEED=SEED)
    
    BATCH_SIZE = 1024
    BUFFER_SIZE = 10000
    
    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
    
    embedding_dim = 128
    
    w2v = Word2Vec(vocab_size, embedding_dim, num_ns)
    w2v.compile(optimizer="adam",
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=["accuracy"])
    
    w2v.fit(dataset, epochs=20)

if(__name__ =="__main__"):
    main()

import tensorflow as tf
import numpy as np
import os
import time
import sys

class GeneratingModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
        self.gru2 = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)
        
    def call(self, inputs, states1=None, states2=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        
        if(states1 is None):
            states1 = self.gru.get_initial_state(x)
            states2 = self.gru2.get_initial_state(x)
        
        x, states1 = self.gru(x, initial_state=states1, training=training)
        x, states2 = self.gru2(x, initial_state=states2, training=training)
        x = self.dense(x, training=training)
        
        if(return_state):
            return x, states1, states2
        else:
            return x
        
class OneStep(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars
        
        # mask to prevent "[unk]" or " " being generated
        skip_ids = self.ids_from_chars(["", "[UNK]"])[:, None]
        sparse_mask = tf.SparseTensor(values=[-float("inf")] * len(skip_ids), 
                                      indices=skip_ids, 
                                      dense_shape=[len(ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)
        
    @tf.function
    def generateOneStep(self, inputs, states=None):
        input_chars = tf.strings.unicode_split(inputs, "UTF-8")
        input_ids = self.ids_from_chars(input_chars).to_tensor()
        
        predicted_logits, states = self.model(inputs=input_ids, states=states, return_state=True)

        # use only last prediction
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits / self.temperature
        # drop score of UNK and "" using mask from earlier
        predicted_logits = predicted_logits + self.prediction_mask
        
        # sample output to generate tokens
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)
        
        # convert ids to chars
        predicted_chars = self.chars_from_ids(predicted_ids)
        
        # return chars and state of model
        return predicted_chars, states

def loadData():
    filepath = tf.keras.utils.get_file("shakespeare.txt", 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
    
    text = open(filepath, "rb").read().decode(encoding="UTF-8")
    
    vocab = sorted(set(text))
    
    return text, vocab

def makeDataset(text, vocab):
    # layer to vectorise characters
    ids_from_chars = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=list(vocab))
    
    # layer to recover characters from vectors
    chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True)
    
    # splitting text into chars
    all_chars = tf.strings.unicode_split(text, input_encoding="UTF-8")
    
    # converting chars to IDs
    all_ids = ids_from_chars(all_chars)
    
    return tf.data.Dataset.from_tensor_slices(all_ids), ids_from_chars, chars_from_ids

def textFromIDs(ids, char_layer):
    
    return tf.strings.reduce_join(char_layer(ids), axis=-1)

def splitInputTarget(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    
    return input_text, target_text

def writeToTxt(fname, result):
    
    f = open(fname, "w+")
    f.write(result)
    f.close()

def main():
    
    text, vocab = loadData()
    
    ids_dataset, char_id_layer, id_char_layer = makeDataset(text, vocab)
    
    seq_length = 100
    sequences = ids_dataset.batch(seq_length + 1, drop_remainder=True)
    
    # for training need set of (input, target) pairs
    # map onto splitting function
    dataset = sequences.map(splitInputTarget)
    
    # split into training batches
    
    BATCH_SIZE = 64
    BUFFER_SIZE = 10000 # instead of shuffling whole set, loads some into buffer of this size and shuffles those
    
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    
    
    # building the model
    
    # vocab length in chars
    vocab_size = len(char_id_layer.get_vocabulary())
    embedding_dim = 256
    # number of rnn units
    rnn_units = 1024
    
    model = GeneratingModel(vocab_size, embedding_dim, rnn_units)
        
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    model.compile(optimizer="adam", loss=loss)
    
    # configuring training checkpoints
    
    checkpoint_dir = "text_gen_checkpoints"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)
    
    if(sys.argv[1] == "train"):
        # training the model
        
        EPOCHS = 30
        history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
    
    elif(sys.argv[1] == "test"):

        model.evaluate(dataset.take(1))
        last_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        model.load_weights(last_checkpoint)
        
        # class for iterating one char at a time
        one_step_model = OneStep(model, id_char_layer, char_id_layer)
        
        start = time.time()
        states1 = None
        states2 = None
        next_char = tf.constant(["ROMEO:"])
        result = [next_char]
        
        for n in range(1000):
            next_char, states1, states2 = one_step_model.generateOneStep(next_char, states1, states2)
            result.append(next_char)
            
        result = tf.strings.join(result)
        
        result = result[0].numpy().decode("utf-8")
        
        writeToTxt("generated_shakespeare.txt", result)
        end = time.time()
        
        print("\nRun time: {}".format(end-start))
    

if(__name__ == "__main__"):
    main()

    
import tensorflow as tf
import kerastuner as kt
import IPython

def getMNIST():
    return tf.keras.datasets.fashion_mnist.load_data()

def normaliseIMG(img):
    return img.astype("float32") / 255     # using numpy's in built vectorisation

class ClearTrainingOutput(tf.keras.callbacks.Callback):
    """
    this clears output at end of each training step 
    """
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait=True)
        
def modelBuilder(hyper_params):
    # this function is passed to the hypermodel builder so it knows loose structure and goals
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    
    # this is tuning the number of neurons in the first dense layer of the model
    # choose optimum between 32 and 512
    hp_units = hyper_params.Int("units", min_value=32, max_value=512, step=32)
    model.add(tf.keras.layers.Dense(units=hp_units, activation="relu"))
    model.add(tf.keras.layers.Dense(10))
    
    # tuning the learning rate 
    # choosing from set of .01, .001, .0001
    hp_learning_rate = hyper_params.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    
    return model
    
def main():
    
    (train_img, train_label), (test_img, test_label) = getMNIST()
    
    train_img = normaliseIMG(train_img)
    test_img = normaliseIMG(test_img)
    
    # building a hypermodel, which has hyper param search space included
    # hyperband is the hyperparam search method, this one is like a championship bracket system
    tuner = kt.Hyperband(modelBuilder,
                         objective="val_accuracy",
                         max_epochs=10,
                         factor=3,
                         directory="KerasTuner",
                         project_name="kt")
    
    # this performs the search of param space
    # if the directories in the above function are full of existing data, it uses that 
    tuner.search(train_img, train_label, 
                 epochs=10, 
                 validation_data=(test_img, test_label), 
                 callbacks=[ClearTrainingOutput()])
    
    best_params = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    model = tuner.hypermodel.build(best_params)
    model.fit(train_img, train_label, epochs=10, validation_data=(test_img, test_label))
    print("Neurons in layer 1: ", best_params.get("units"))
    print("Learning Rate: ", best_params.get("learning_rate"))
    

if(__name__ == "__main__"):
    main()

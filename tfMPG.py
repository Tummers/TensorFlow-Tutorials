import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def readData():
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    #filepath = "C:\\Users\\tomjs\\Documents\\Python Scripts\\TensorFlow\\MPG"
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 
                    'Acceleration', 'Model Year', 'Origin']
    raw_data = pd.read_csv(url, names=column_names, na_values='?', comment='\t',
                           sep=' ', skipinitialspace=True)
    
    return raw_data

def splitTrainTest(data, fraction_train):
    training  = data.sample(frac=fraction_train, random_state=0)
    testing = data.drop(training.index)
    
    return training, testing

def gridPlot(data):
    sns.pairplot(data[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind="kde")
    plt.show()

def lossPlot(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.ylim([0, 10])
    plt.xlabel("Epoch")
    plt.ylabel("Error [MPG]")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    raw_data = readData()
    dataset = raw_data.copy()
    
    dataset = dataset.dropna() # removes unknown values
    
    # changing numeric values for us jap and eu to true or false across three categories
    dataset["Origin"] = dataset["Origin"].map({1: "USA", 2: "Europe", 3: "Japan"})
    dataset = pd.get_dummies(dataset, prefix=" ", prefix_sep=" ")
    
    # split the training and testing set
    train_dataset, test_dataset = splitTrainTest(dataset, 0.8)
    
    # split the target from the data
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()
    
    train_labels = train_features.pop('MPG')
    test_labels = test_features.pop('MPG')
    
    # make normalisation layer
    normaliser = tf.keras.layers.experimental.preprocessing.Normalization()
    normaliser.adapt(np.array(train_features))
    
    model = tf.keras.Sequential([normaliser, 
                                 tf.keras.layers.Dense(64, activation="relu"),
                                 tf.keras.layers.Dense(64, activation="relu"),
                                 tf.keras.layers.Dense(units=1)])

    model.compile(loss="mean_absolute_error",
                         optimizer=tf.optimizers.Adam(learning_rate=0.001))
    
    history = model.fit(train_features, train_labels,
                        epochs=100,
                        verbose=0,
                        validation_split=0.2)
   
    lossPlot(history)
    
if(__name__ == "__main__"):
    main()


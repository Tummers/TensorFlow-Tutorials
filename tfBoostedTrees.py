import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def loadData():
    train = pd.read_csv('C:\\Users\\tomjs\\Documents\\Python Scripts\\TensorFlow\\Titanic\\train.csv')
    test = pd.read_csv('C:\\Users\\tomjs\\Documents\\Python Scripts\\TensorFlow\\Titanic\\eval.csv')
    train_label = train.pop("survived")
    test_label = test.pop("survived")
    
    return train, train_label, test, test_label

def one_hot_categorical_column(feature_name, vocab):
    # turns categorical columns into one-hot, i.e. class becomes a list of three, third class passenger is [0, 0, 1]
    return tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocab))

def makeInputFunction(X, y, no_epochs=None, shuffle=True):
    def inputFunc():
        dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
        if(shuffle == True):
            dataset = dataset.shuffle(len(X)) # typically wouldn'y use whole set, but dataset is small here
            
        dataset = dataset.repeat(no_epochs)
        
        dataset = dataset.batch(len(X))
        return dataset
    return inputFunc

def _getColour(value):
    """
    makes positive and negative contributions green and red
    """
    green, red = sns.color_palette()[2:4]
    
    if(value >= 0):
        return green
    return red
    
def addFeatureValues(feature_values, ax):
    """
    display value on left side of plot
    """
    x_coord = ax.get_xlim()[0] # get far left x
    offset = 0.15
    
    for y_coord, (feat_name, feat_val) in enumerate(feature_values.items()):
        string = plt.text(x_coord, y_coord - offset, "{}".format(feat_val), size=10)
        string.set_bbox(dict(facecolor="white", alpha=0.5))
        
    string = plt.text(x_coord, y_coord + 1 - offset, "Feature\nValue", size=12)
        
def plotExample(dataframe_dfc, dataframe_testing, ID):
    top_n = 8 # show top 8 contributions
    # sort contributions by magnitude
    example = dataframe_dfc.iloc[ID]
    sorted_indices = example.abs().sort_values()[-top_n:].index
    example = example[sorted_indices]
    colours = example.map(_getColour).tolist()
    
    ax = example.to_frame().plot(kind="barh", color=[colours], legend=None, alpha=.75, figsize=(10, 6))
    ax.grid(False, axis="y")
    ax.set_yticklabels(ax.get_yticklabels(), size=14)
    
    addFeatureValues(dataframe_testing.iloc[ID][sorted_indices], ax)
    
    return ax
    
def main():
    train, train_label, test, test_label = loadData()
    tf.random.set_seed(123) # means results are random but consistant
    
    CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                           'embark_town', 'alone']
    NUMERICAL_COLUMNS = ['age', 'fare']

    # this bit is converting categories into numbers, and joining with numerical ones
    feature_columns = []
    for feature_name in CATEGORICAL_COLUMNS:
        vocabulary = train[feature_name].unique()
        feature_columns.append(one_hot_categorical_column(feature_name, vocabulary))
        
    for feature_name in NUMERICAL_COLUMNS:
        feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

    training_input = makeInputFunction(train, train_label)
    testing_input = makeInputFunction(test, test_label, shuffle=False, no_epochs=1)
    
    # to get descriptions you need centre bias on
    params = {
      'n_trees': 50,
      'max_depth': 3,
      'n_batches_per_layer': 1,
      # You must enable center_bias = True to get DFCs. This will force the model to
      # make an initial prediction before using any features (e.g. use the mean of
      # the training labels for regression or log odds for classification when
      # using cross entropy loss).
      'center_bias': True
    }
    
    est = tf.estimator.BoostedTreesClassifier(feature_columns, **params)
    
    est.train(training_input, max_steps=100)
    result = est.evaluate(testing_input)
    pd.Series(result).to_frame()
    #print(pd.Series(result))
    
    #pred_dicts = list(est.predict(testing_input))
    #probs = pd.Series([pred["probabilities"][1] for pred in pred_dicts])
    
    """
    probs.plot(kind="hist", bins=20)
    plt.show()
    """
    explained_pred_dicts = list(est.experimental_predict_with_explanations(testing_input))
    explained_probs = pd.Series([pred["probabilities"][1] for pred in explained_pred_dicts])
    
    labels = test_label.values
    df_dfc = pd.DataFrame([pred["dfc"] for pred in explained_pred_dicts])
    df_dfc.describe().T
    
    # the prob associated with each item is the sum of dfcs + bias
    bias = explained_pred_dicts[0]["bias"]
    dfc_probability = df_dfc.sum(axis=1) + bias
    np.testing.assert_almost_equal(dfc_probability.values, explained_probs.values)
    
    #plotting 
    ID = 182
    ax = plotExample(df_dfc, test, ID)
    ax.set_title('Feature contributions for example {}\n pred: {:1.2f}; label: {}'.format(ID, explained_probs[ID], labels[ID]))
    ax.set_xlabel('Contribution to predicted probability', size=14)

    plt.show()
    
if(__name__ == "__main__"):
    main()
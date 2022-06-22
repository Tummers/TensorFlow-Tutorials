import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow as tf
import tensorflow.compat.v2.feature_column as fc

def loadData():
    dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
    dfevaluation = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
    y_train = dftrain.pop('survived')
    y_evaluation = dfevaluation.pop('survived')
    
    return dftrain, dfevaluation, y_train, y_evaluation

def makeInputFunction(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def inputFunction():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if(shuffle == True):
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return inputFunction

def main():
    
    train, test, train_label, test_label = loadData()
    
    CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                           'embark_town', 'alone']
    NUMERIC_COLUMNS = ['age', 'fare']
    
    feature_columns = []
    for feature_name in CATEGORICAL_COLUMNS:
        vocabulary = train[feature_name].unique() # gives each unique classification a value
        feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
        
    for feature_name in NUMERIC_COLUMNS:
        feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

    train_input_func = makeInputFunction(train, train_label)
    test_input_func = makeInputFunction(test, test_label, num_epochs=1, shuffle=False)
    """
    linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
    linear_est.train(train_input_func)
    result = linear_est.evaluate(test_input_func)
    print(result)
    """
    """
    including derived features
    """
    age_x_gender = tf.feature_column.crossed_column(['age', 'sex'], hash_bucket_size=100)
    derived_feature_columns = [age_x_gender]
    
    linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns+derived_feature_columns)
    linear_est.train(train_input_func)
    result = linear_est.evaluate(test_input_func)
    print(result)
    
if(__name__=="__main__"):
    main()


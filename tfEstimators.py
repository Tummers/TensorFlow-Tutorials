import tensorflow as tf
import pandas as pd

def readData():
    CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
    SPECIES = ['Setosa', 'Versicolor', 'Virginica']
    
    train_path = tf.keras.utils.get_file("iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
    test_path = tf.keras.utils.get_file("iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")
    
    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

    return train, test

def inputFunction(features, labels, training=True, batch_size=256):
    """
    This is the input function for the estimator,
    it returns a tuple,
    the first element is a dict of feature names, combined with an array of all the values for the feature,
    the second element is an array of target labels for all the data values
    """
    # first converting the features and labels into a pandas Dataset object
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    
    # when training you shuffle the data and repeat 
    if(training == True):
        dataset = dataset.shuffle(1000).repeat()
    
    # return batch_size elements of dataset for this epoch of training
    return dataset.batch(batch_size)

def inputForPrediction(features, batch_size=256):
    # convert to label-less dataset
    
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)
    

def main():
    # reading the data in from iris dataset online
    train, test = readData()
    SPECIES = ['Setosa', 'Versicolor', 'Virginica']
    
    # seperating the target values
    train_target = train.pop("Species")
    test_target = test.pop("Species")
    
    # defining the feature columns
    feature_columns = []
    for key in train.keys():
        feature_columns.append(tf.feature_column.numeric_column(key=key))
        
    # instantiating the estimator with a DNN classifier
    
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[30, 10],
                                            n_classes=3)
    
    # by wrapping the input function in a lambda you can provide arguments to a function, estimator expects no arguments in the input function
    classifier.train(input_fn=lambda: inputFunction(train, train_target, training=True),
                     steps=5000)
    
    result = classifier.evaluate(input_fn=lambda: inputFunction(test, test_target, training=False))
    
    print("Accuracy of Model: ", result["accuracy"])
    
    expected = ["Setosa", "Versicolor", "Virginica"]
    predict_input = {'SepalLength': [5.1, 5.9, 6.9],
                     'SepalWidth': [3.3, 3.0, 3.1],
                     'PetalLength': [1.7, 4.2, 5.4],
                     'PetalWidth': [0.5, 1.5, 2.1]}
    
    predictions = classifier.predict(input_fn=lambda: inputForPrediction(predict_input))
    
    for predicted_dict, expected_outcome in zip(predictions, expected):
        class_id = predicted_dict["class_ids"][0]
        prob = predicted_dict["probabilities"][class_id]
        
        print("Prediction: '{}' ({:.2f}%). Expectation '{}'".format(SPECIES[class_id], 100 * prob, expected_outcome))
    
if(__name__ == "__main__"):
    main()
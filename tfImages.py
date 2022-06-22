import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def getData():
    return tf.keras.datasets.cifar10.load_data()

def plotImagesWithLabels(train_images, train_labels, n):
    """
    shows the first n images in the dataset with their labels
    """
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    plt.figure(figsize=(10, 10))
    rows_and_columns = int(np.sqrt(n))
    for i in range(n):
        plt.subplot(rows_and_columns, rows_and_columns, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i][0]])

    plt.show()
    
def buildConvModel(dimension_1, dimension_2):
    """
    builds the convolutional section of the model
    """
    conv_kernel_size = (3, 3)
    filter_no_first = 32
    pool_size = (2, 2)
    filter_no_second = 64
    filter_no_third = 64
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filter_no_first, conv_kernel_size, activation="relu", input_shape=(dimension_1, dimension_2, 3)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size))
    
    model.add(tf.keras.layers.Conv2D(filter_no_second, conv_kernel_size, activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size))
    
    model.add(tf.keras.layers.Conv2D(filter_no_third, conv_kernel_size, activation="relu"))
    
    print(model.summary())
    
    return model

def addDenseBase(model):
    """
    adds the final output layer of the model, 
    reduces from size of final layer to size 10 using dense connection
    """
    output_shape = model.layers[-1].filters
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(output_shape, activation="relu"))
    model.add(tf.keras.layers.Dense(10))
    
    return model

def plotTogether(fit, label1, label2):
    
    plt.plot(fit.history[label1], label=label1)
    plt.plot(fit.history[label2], label=label2)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    
    min1 = np.min(fit.history[label1])
    min2 = np.min(fit.history[label2])
    maxi = 1
    
    plt.ylim([min(min1, min2), maxi])
    plt.show()
    
def main():
    # get dataset
    (train_images, train_labels), (test_images, test_labels) = getData()
    
    # normalise image data to between 0 and 1
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    
    #plotImagesWithLabels(train_images, train_labels, 25)
    
    # getting image size
    image_dim_1 = train_images[0].shape[0]
    image_dim_2 = train_images[0].shape[0]
    
    # build conv model
    model = buildConvModel(image_dim_1, image_dim_2)
    
    model = addDenseBase(model)
    
    print(model.summary())
    
    #compiling the model is next

    model.compile(optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    
    history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
    
    plotTogether(history, "accuracy", "val_accuracy")

if(__name__ == "__main__"):
    main()
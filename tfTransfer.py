import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

def loadData(batch_size, img_size):
    url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
    zip_path = tf.keras.utils.get_file("cats_and_dogs.zip", origin=url, extract=True)
    PATH = os.path.join(os.path.dirname(zip_path), 'cats_and_dogs_filtered')

    train_dir = os.path.join(PATH, "train")
    val_dir = os.path.join(PATH, "validation")
    
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                   shuffle=True,
                                                                   batch_size=batch_size,
                                                                   image_size=img_size)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(val_dir,
                                                                   shuffle=True,
                                                                   batch_size=batch_size,
                                                                   image_size=img_size)

    return train_ds, val_ds

def valTestSplit(val_set, proportion):
    
    val_batches = tf.data.experimental.cardinality(val_set)
    test_set = val_set.take(val_batches // int(1 / proportion))
    val_set = val_set.skip(val_batches // int(1 / proportion))
    
    return val_set, test_set

def augmentationLayer():
    
    layer = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)])
    return layer

def buildModel(train_set, img_size):
    
    img_shape = img_size + (3,)
    
    # this takes an existing model which can do lots of classifying
    base_model = tf.keras.applications.MobileNetV2(input_shape=img_shape,
                                                   include_top=False,
                                                   weights="imagenet")
    
    image_batch, label_batch = next(iter(train_set))
    feature_batch = base_model(image_batch)

    # going to use the base as a feature detector and add a classifier to that
    base_model.trainable = False # this freezes the base so it isn't changed by training
    
    # this is the part where we tune the model to classify our images,
    # we add an averaging layer to process the final layer of features
    
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    
    # classification layer is single output + means class 1, - means class 2
    prediction_layer = tf.keras.layers.Dense(1)
    prediction_batch = prediction_layer(feature_batch_average)
    
    model = tf.keras.Sequential([tf.keras.Input(shape=img_shape),
                                 base_model, 
                                 global_average_layer,
                                 tf.keras.layers.Dropout(0.2),
                                 prediction_layer
                                 ])
    
    return model

def showNine(dataset):
    
    plt.figure(figsize=(10, 10))

    for images, labels in dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(dataset.class_names[labels[i]])
            plt.axis("off")
            
    plt.show()
    
def plotLossAccuracy(history):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    
    epochs = np.arange(len(acc))
    
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label="training accuracy")
    plt.plot(epochs, val_acc, label="validation accuracy")
    plt.legend(loc="lower right")
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label="training loss")
    plt.plot(epochs, val_loss, label="validation loss")
    plt.legend(loc="upper right")
    
    plt.show()

def main():
    batch_size = 32
    img_size = (160, 160)

    train, val = loadData(batch_size, img_size)
    #showNine(train)
    
    val, test = valTestSplit(val, 0.2)

    # preloading data into buffer
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train = train.prefetch(buffer_size=AUTOTUNE)
    val = val.prefetch(buffer_size=AUTOTUNE)
    test = test.prefetch(buffer_size=AUTOTUNE)
    
    aug_layer = augmentationLayer()
    # layer to rescale to range -1 -> 1
    rescaling_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1)
    
    model = buildModel(train, img_size)
    
    model = tf.keras.Sequential([tf.keras.Input(shape=img_size + (3,)),
                                 aug_layer,
                                 rescaling_layer,
                                 model])
    
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                  loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    
    print(model.summary())
    
    epochs = 10
    init_loss, init_acc = model.evaluate(val)
    print("Init Loss: {:.2f}\nInit Acc: {:.2f}".format(init_loss, init_acc))
    
    history = model.fit(train,
                        epochs=epochs,
                        validation_data=val)
    
    plotLossAccuracy(history)
    
    # next comes the fine tuning of the model,
    # having done some training of our own final layers,
    # we now do some training of the last couple layers of the base model
    
    
    
if(__name__ == "__main__"):
    main()
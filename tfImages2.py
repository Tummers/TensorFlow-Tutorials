import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import PIL # python image library?

def downloadData():
    """
    gets the dataset off the internet and returns the directory it is put in
    """
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)
    
    return data_dir

def exploreData(data_dir):
    """
    poking around at number of images/looking at some
    """
    roses = list(data_dir.glob("roses/*"))
    rose = PIL.Image.open(str(roses[1]))
    rose.show()
    
def imgsWithLabels(dataset, n):
    """
    shows n images with their corresponding labels
    """
    class_names = dataset.class_names
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(n):
            ax = plt.subplot(3, 3, i+1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
            
    plt.show()
    
def createDataset(data_dir, img_size, batch_size):
    """
    creating a keras dataset
    """
    img_height = img_size[0]
    img_width = img_size[1]
    
    train_set = tf.keras.preprocessing.image_dataset_from_directory(data_dir,
                                                                    validation_split=0.2,
                                                                    subset="training",
                                                                    seed=123,
                                                                    image_size=(img_height, img_width),
                                                                    batch_size=batch_size)
    val_set = tf.keras.preprocessing.image_dataset_from_directory(data_dir,
                                                                  validation_split=0.2,
                                                                  subset="validation",
                                                                  seed=123,
                                                                  image_size=(img_height, img_width),
                                                                  batch_size=batch_size)
    
    #imgsWithLabels(train_set, 9)
    
    return train_set, val_set

def buildModel(img_size):
    """
    builds a keras model
    """
    img_height = img_size[0]
    img_width = img_size[1]
    num_classes = 5
    
    augmentation_layer = tf.keras.models.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.1)])
    
    normalisation_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    
    model = tf.keras.models.Sequential([augmentation_layer,
                                        normalisation_layer,
                                        tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
                                        tf.keras.layers.MaxPooling2D(),
                                        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
                                        tf.keras.layers.MaxPooling2D(),
                                        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
                                        tf.keras.layers.MaxPooling2D(),
                                        tf.keras.layers.Dropout(0.2),
                                        tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(128, activation="relu"),
                                        tf.keras.layers.Dense(num_classes)])
    
    return model

def plotAccuracy(history):
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
    data_dir = downloadData()
    #exploreData(data_dir)
    img_size = (180, 180)
    batch_size = 32
    train, val = createDataset(data_dir, img_size, batch_size)
    
    # loading the data into memory so that training isn't slowed down,
    # by constantly fetching data
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    
    train = train.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val = val.cache().prefetch(buffer_size=AUTOTUNE)
    
    model = buildModel(img_size)
    model.compile(optimizer="adam",
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    
    print(model.summary())
    
    epochs = 10
    history = model.fit(train, validation_data=val, epochs=epochs)
    
    plotAccuracy(history)
    
if(__name__ == "__main__"):
    main()

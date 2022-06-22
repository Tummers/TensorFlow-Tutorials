import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def readData(fname):
    
    raw_data = np.loadtxt(fname, delimiter=",")
    
    labels = raw_data[:, 0]
    images_raw = raw_data[:, 1:]
    images = np.empty([raw_data.shape[0], 28, 28])
    for i in range(images_raw.shape[0]):
        images[i] = images_raw[i].reshape([28, 28])
        
    return images, labels

def plot_image(image, prediction, label):
    
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image, cmap="gray")
    
    guessed_label = np.argmax(prediction)
    
    if(guessed_label == label):
        colour = "g"
    else:
        colour = "r"
    
    plt.xlabel("{} {:2.0f}% ({})".format(guessed_label,
                                         100*np.max(prediction),
                                         label,),
               color=colour)
    
def plot_certainty(prediction, label):
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    plot = plt.bar(range(10), prediction, color="gray")
    plt.ylim([0, 1])
    guessed_label = np.argmax(prediction)
    
    plot[guessed_label].set_color("red")
    plot[label].set_color("green")
    
def main():
    
    training_fname = "mnist_train_reduced10.csv"
    testing_fname = "mnist_test_reduced10.csv"
    
    training_images, training_labels = readData(training_fname)
    testing_images, testing_labels = readData(testing_fname)
    
    model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                                 tf.keras.layers.Dense(100, activation="relu"),
                                 tf.keras.layers.Dense(10)
                                ])
    
    model.compile(optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    
    model.fit(training_images, training_labels, epochs=10)
    

    test_loss, test_acc = model.evaluate(testing_images, testing_labels, verbose=2)
    print("\n\nTest Accuracy: ", test_acc)
    
    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])
    
    predictions = probability_model.predict(testing_images)

    show_no = 5
    for i in range(show_no):
        rand_index = int(np.random.uniform(0, testing_images.shape[0]))
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plot_image(testing_images[rand_index], predictions[rand_index], testing_labels[rand_index])
        plt.subplot(1, 2, 2)

        plot_certainty(predictions[rand_index], int(testing_labels[rand_index]))
        plt.show()
    
if(__name__ == "__main__"):
    main()
import numpy as np
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt

# Funkcija za prikaz nekoliko loše klasificiranih slika
def prikazi_greske(x_test, y_test, predictions, num_images=5):
    misclassified_indexes = np.where(np.argmax(y_test, axis=1) != np.argmax(predictions, axis=1))[0]
    selected_indexes = misclassified_indexes[:num_images]

    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(selected_indexes):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f"Stvarna: {np.argmax(y_test[idx])}\nPredviđena: {np.argmax(predictions[idx])}")
        plt.axis('off')
    plt.show()

# Model / podaci
num_classes = 10
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Skaliranje slika
x_test = x_test.astype("float32") / 255
x_test = np.expand_dims(x_test, -1)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Učitavanje modela
model = keras.models.load_model("LV8/model_1zad.keras")

# Predviđanje na test skupu
predictions = model.predict(x_test)

# Prikaz nekoliko loše klasificiranih slika
prikazi_greske(x_test, y_test, predictions)

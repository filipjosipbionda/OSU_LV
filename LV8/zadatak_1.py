    import numpy as np
    from tensorflow import keras
    from keras import layers
    import seaborn as sns
    from matplotlib import pyplot as plt
    from sklearn.metrics import confusion_matrix

    # Model / data parameters
    num_classes = 10  # Promijenjeno na 10 jer imamo 10 klasa u MNIST skupu podataka
    input_shape = (28, 28, 1)

    # Učitavanje train i test podataka
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Prikaz karakteristika train i test podataka
    print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
    print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

    # Prikaz nekoliko slika iz train skupa
    plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(x_train[i])
        plt.xlabel(y_train[i])
    plt.show()

    # Skaliranje slika na raspon [0,1]
    x_train_s = x_train.astype("float32") / 255
    x_test_s = x_test.astype("float32") / 255

    # Slike trebaju biti (28, 28, 1)
    x_train_s = np.expand_dims(x_train_s, -1)
    x_test_s = np.expand_dims(x_test_s, -1)

    print("x_train shape:", x_train_s.shape)
    print(x_train_s.shape[0], "train samples")
    print(x_test_s.shape[0], "test samples")

    # Pretvorba oznaka u jedno-hot kodirane vektore
    y_train_s = keras.utils.to_categorical(y_train, num_classes)
    y_test_s = keras.utils.to_categorical(y_test, num_classes)

    # Kreiranje modela
    model = keras.Sequential()
    model.add(layers.Input(shape=(28,28,1)))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation="relu"))
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dense(num_classes, activation="softmax"))  # Broj izlaznih jedinica postavljen na num_classes

    # Definiranje karakteristika procesa učenja
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Provedba učenja mreže
    batch_size = 32
    epochs = 10
    history = model.fit(x_train_s, y_train_s, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    predictions = model.predict(x_test_s)

    # Prikaz test accuracy i matrice zabune
    score = model.evaluate(x_test_s, y_test_s, verbose=0)
    conf_matrix = confusion_matrix(np.argmax(y_test_s, axis=1), np.argmax(predictions, axis=1))
    print(conf_matrix)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # Spremanje modela
    model.save("model_1zad.keras")

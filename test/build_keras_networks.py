from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#* Fashion mnist nn

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = tf.reshape(train_images, [-1, 28, 28, 1])
test_images = tf.reshape(test_images, [-1, 28, 28, 1])

model = keras.Sequential([
    keras.layers.AveragePooling2D(input_shape=(28, 28, 1), pool_size=(2,2),strides=(2,2), padding='valid', data_format='channels_last'),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

model.save('fashion_mnist.h5')

#predictions = model.predict(test_images)
#print(predictions)
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# TODO: IMPORT DATA FROM CSV INSTEAD OF KERAS
# Loading mnist dataset
data = tf.keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = data.load_data()

# Normalize the data
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

# Create the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=1)

numRun = 2 # Number of times the model has been run
model.save('../Using Libraries/output/model' + str(numRun) + '.model')

loss, accuracy = model.evaluate(X_test, Y_test)

print("Loss: ", loss)
print("Accuracy: ", accuracy)
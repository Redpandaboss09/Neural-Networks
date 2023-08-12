import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import model_functions as mf
import prediction_functions as pf
import serialization as sr

run_count = sr.get_run_count() # Number of times the model has been run (for serialization)

# Importing the dataset and converting it into numpy array
data = pd.read_csv('../Data/train.csv')
data = np.array(data)

# Get dimensions of the data and shuffle it
m, n = data.shape
np.random.shuffle(data)

# Split the data for training and testing
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_, m_train = X_train.shape

# Train the model
W1, b1, W2, b2, W3, b3, losses, accuracy = mf.gradient_descent(X_train, Y_train, 500, 0.4, m)

# Plot the losses
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

# Test the model / See results
pf.test_prediction(X_train, Y_train, 4, W1, b1, W2, b2, W3, b3)
pf.test_prediction(X_train, Y_train, 42, W1, b1, W2, b2, W3, b3)
pf.test_prediction(X_train, Y_train, 10, W1, b1, W2, b2, W3, b3)
pf.test_prediction(X_train, Y_train, 6, W1, b1, W2, b2, W3, b3)

# Test the model on the dev set
# predictions = pf.make_prediction(X_dev, W1, b1, W2, b2, W3, b3)
# accuracy = mf.get_accuracy(predictions, Y_dev)

# Save the model and other information
params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
cur_dir = sr.create_output_dir("model_" + str(run_count))

sr.save_model(params, cur_dir + "/model_" + str(run_count) + ".pkl")
sr.save_training_info(losses, accuracy, cur_dir + "/training_info_" + str(run_count) + ".pkl")
sr.save_loss_chart(losses, cur_dir + "/loss_chart_" + str(run_count) + ".png")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import model_functions as mf
import prediction_functions as pf
import serialization as sr

run_count = sr.get_run_count() # Number of times the model has been run (for serialization)

# Importing the datasets and converting it into numpy array
dataMNIST = pd.read_csv("../Data/emnist-mnist-train.csv")
dataEMNIST = pd.read_csv("../Data/emnist-digits-train.csv")

dataMNIST.columns = dataEMNIST.columns

data = pd.concat([dataMNIST, dataEMNIST])
data = np.array(data)

# Get dimensions of the data and shuffle it
m, n = data.shape
np.random.shuffle(data)

# Split the data for training, testing, and validation
data_dev = data[0:30000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_val = data[30000:60000].T
Y_val = data_val[0]
X_val = data_val[1:n]
X_val = X_val / 255.

data_train = data[60000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_, m_train = X_train.shape

# Train the model
W1, b1, W2, b2, losses, accuracy, accuracy_plot, time = mf.gradient_descent(X_train, Y_train, X_val, Y_val, 500, 0.4, m)

# Plot the losses
plt.plot(losses[0], label='Training loss')
plt.plot(losses[1], label='Validation loss')
plt.title("Loss Curves")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')
plt.show()

# Plot the accuracy
plt.plot(accuracy_plot[0], label='Training Accuracy')
plt.plot(accuracy_plot[1], label='Validation Accuracy')
plt.title("Accuracy Curve")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='upper right')
plt.show()

# Test the model / See results
pf.test_prediction(X_train, Y_train, 4, W1, b1, W2, b2)
pf.test_prediction(X_train, Y_train, 42, W1, b1, W2, b2)
pf.test_prediction(X_train, Y_train, 10, W1, b1, W2, b2)
pf.test_prediction(X_train, Y_train, 6, W1, b1, W2, b2)

# Test the model on the dev set
predictions = pf.make_prediction(X_dev, W1, b1, W2, b2)
accuracy_validation = mf.get_accuracy(predictions, Y_dev)

print(predictions)
print(accuracy_validation)

# Save the model and other information
params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
cur_dir = sr.create_output_dir("model_" + str(run_count))

sr.save_model(params, cur_dir + "/model_" + str(run_count) + ".pkl")
sr.save_training_info(losses, accuracy, time, cur_dir + "/training_info_" + str(run_count) + ".pkl")
sr.save_loss_chart(losses, cur_dir + "/loss_chart_" + str(run_count) + ".png")
sr.save_accuracy_chart(accuracy_plot, cur_dir + "/accuracy_chart_" + str(run_count) + ".png")
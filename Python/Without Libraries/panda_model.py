import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import model_functions as mf
import prediction_functions as pf

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
W1, b1, W2, b2 = mf.gradient_descent(X_train, Y_train, 500, 0.1, m)

# Test the model / See results
pf.test_prediction(X_train, Y_train, 4, W1, b1, W2, b2)
import numpy as np
import time

# Initial parameters
def init_parameters():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return W1, b1, W2, b2

# ReLu and Softmax activation functions
def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    return np.exp(Z) / sum(np.exp(Z))

# Derivative functions
def dReLU(Z):
    return Z > 0

# Encode the labels (One hot encoding)
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T

    return one_hot_Y

# Forward and backward propagation
def forward_propagation(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2

def backward_propagation(Z1, A1, Z2, A2, W1, W2, X, Y, m):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1).reshape(-1, 1)
    dZ1 = W2.T.dot(dZ2) * dReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1).reshape(-1, 1)


    return dW1, db1, dW2, db2

# Update parameters
def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * np.reshape(db1, (10, 1))
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * np.reshape(db2, (10, 1))

    return W1, b1, W2, b2

# Get accuracy and predictions for model loop
def get_accuracy(predictions, Y):
    # print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def get_predictions(A2):
    return np.argmax(A2, 0)

# Gradient descent for training
def gradient_descent(X_train, Y_train, X_val, Y_val, iterations, alpha, m):
    W1, b1, W2, b2, = init_parameters()
    losses = [] # For plotting loss on test set
    val_losses = [] # For plotting loss on validation set
    accuracy_train = [] # For plotting accuracy on test set
    accuracy_val = [] # For plotting accuracy on validation set

    start_time = time.time() # For timing the training

    for i in range(iterations):
        Z1, A1, Z2, A2, = forward_propagation(W1, b1, W2, b2, X_train)
        dW1, db1, dW2, db2 = backward_propagation(Z1, A1, Z2, A2, W1, W2, X_train, Y_train, m)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        # Calculate loss on test set
        loss = -np.sum(np.log(A2) * one_hot(Y_train)) / m
        losses.append(loss)

        # Calculate loss on validation set
        _, _, _, A_val = forward_propagation(W1, b1, W2, b2, X_val)
        loss_val = -np.sum(np.log(A_val) * one_hot(Y_val)) / m
        val_losses.append(loss_val)

        if i % 10 == 0:
            # Calculate training accuracy
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            accuracy_iteration_train = get_accuracy(predictions, Y_train)
            accuracy_train.append(accuracy_iteration_train)
            print("Accuracy: ", accuracy_iteration_train)

            # Calculate validation accuracy
            predictions_val = get_predictions(A_val)
            accuracy_iteration_val = get_accuracy(predictions_val, Y_val)
            accuracy_val.append(accuracy_iteration_val)

        # Final accuracy
        accuracy = get_accuracy(predictions, Y_train)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time)

    return W1, b1, W2, b2, (losses, val_losses), accuracy, (accuracy_train, accuracy_val), elapsed_time

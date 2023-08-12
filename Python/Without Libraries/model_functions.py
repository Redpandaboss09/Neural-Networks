import numpy as np

# Initial parameters
def init_parameters():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    W3 = np.random.rand(10, 10) - 0.5
    b3 = np.random.rand(10, 1) - 0.5

    return W1, b1, W2, b2, W3, b3

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
def forward_propagation(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)

    return Z1, A1, Z2, A2, Z3, A3

def backward_propagation(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y, m):
    one_hot_Y = one_hot(Y)
    dZ3 = A3 - one_hot_Y
    dW3 = 1 / m * dZ3.dot(A2.T)
    db3 = 1 / m * np.sum(dZ3, axis=1).reshape(-1, 1)
    dZ2 = W3.T.dot(dZ3) * dReLU(Z2)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1).reshape(-1, 1)
    dZ1 = W2.T.dot(dZ2) * dReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1).reshape(-1, 1)


    return dW1, db1, dW2, db2, dW3, db3

# Update parameters
def update_parameters(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * np.reshape(db1, (10, 1))
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * np.reshape(db2, (10, 1))
    W3 = W3 - alpha * dW3
    b3 = b3 - alpha * np.reshape(db3, (10, 1))

    return W1, b1, W2, b2, W3, b3

# Get accuracy and predictions for model loop
def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def get_predictions(A3):
    return np.argmax(A3, 0)

# Gradient descent for training
def gradient_descent(X, Y, iterations, alpha, m):
    W1, b1, W2, b2, W3, b3 = init_parameters()
    losses = [] # For plotting

    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_propagation(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = backward_propagation(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y, m)
        W1, b1, W2, b2, W3, b3 = update_parameters(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)

        #Calculate loss
        loss = -np.sum(np.log(A3) * one_hot(Y)) / m
        losses.append(loss)

        if i % 10 == 0:
            print("Iteration: ", i)
            print("Loss: ", loss)
            predictions = get_predictions(A3)
            print("Accuracy: ", get_accuracy(predictions, Y))

    return W1, b1, W2, b2, W3, b3, losses

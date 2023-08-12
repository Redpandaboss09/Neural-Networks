import numpy as np
import matplotlib.pyplot as plt
import model_functions as mf

# Following methods are for testing / seeing the results of the model
def make_prediction(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = mf.forward_propagation(W1, b1, W2, b2, W3, b3, X)
    predictions = mf.get_predictions(A3)
    return predictions

def test_prediction(X_train, Y_train, index, W1, b1, W2, b2, W3, b3):
    cur_image = X_train[:, index, None]
    prediction = make_prediction(X_train[:, index, None], W1, b1, W2, b2, W3, b3)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    cur_image = cur_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(cur_image, interpolation='nearest')
    plt.show()

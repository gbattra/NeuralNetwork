import numpy as np
from sigmoid import sigmoid


def predict(X, W):

    # unwrap Thetas
    Theta_1, Theta_2 = W

    # add 1s to input layer and make predictions for input layer
    a1_0 = np.ones((X.shape[0], 1))
    a1 = np.hstack((a1_0, X))
    z2 = a1.dot(Theta_1.T)

    # get second layer inputs, add 1s and make predictions
    a2_0 = np.ones((z2.shape[0], 1))
    a2 = np.hstack((a2_0, sigmoid(z2)))
    z3 = a2.dot(Theta_2.T)
    h = sigmoid(z3)

    return h

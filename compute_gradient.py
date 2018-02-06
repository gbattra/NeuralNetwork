import numpy as np
from sigmoid import sigmoid


def compute_gradient(W, L_in, hidden_layer, L_out, X, y, L):
    # unwrap W into initial thetas
    Theta_1_size = hidden_layer * (L_in + 1)
    Theta_2_size = L_out * (hidden_layer + 1)
    Theta_1_flat = W[0, 0:Theta_1_size]
    Theta_2_flat = W[0, Theta_1_size:(Theta_1_size + Theta_2_size)]

    # reshape W into Thetas
    Theta_1 = Theta_1_flat.reshape(hidden_layer, L_in + 1)
    Theta_2 = Theta_2_flat.reshape(L_out, hidden_layer + 1)

    # initialize temp Thetas w/out ones column
    temp_Theta_1 = Theta_1.copy()
    temp_Theta_2 = Theta_2.copy()
    temp_Theta_1[:, 0] = 0
    temp_Theta_2[:, 0] = 0

    # store training size
    m = X.shape[0]

    # add 1s to input layer and make predictions for input layer
    a1_0 = np.ones((X.shape[0], 1))
    a1 = np.hstack((a1_0, X))
    z2 = a1 * Theta_1.T

    # get second layer inputs, add 1s and make final layer predictions
    a2_0 = np.ones((z2.shape[0], 1))
    a2 = np.hstack((a2_0, sigmoid(z2)))
    z3 = a2 * Theta_2.T
    h = sigmoid(z3)

    # perform backpropagation to compute gradients
    d3 = h - y
    d2 = np.multiply(d3 * Theta_2, np.multiply(a2, 1 - a2))
    d2 = d2[:, 1:]  # remove 1s column from delta
    Theta_2_grad = (d3.T * a2) / m
    Theta_1_grad = (d2.T * a1) / m

    # add regularization term
    Theta_1_grad += (L / m) * temp_Theta_1
    Theta_2_grad += (L / m) * temp_Theta_2

    # unroll gradients
    grad = np.hstack((Theta_1_grad.flatten(), Theta_2_grad.flatten()))

    return grad

# compute the cost for the neural network
# returns J and grad for a given Theta set

import numpy as np
from sigmoid import sigmoid


def compute_cost(W, L_in, hidden_layer, L_out, X, y, L):

    # unwrap W into initial thetas
    Theta_1_size = (hidden_layer * (L_in + 1))
    Theta_2_size = (L_out * (hidden_layer + 1))
    Theta_1_flat = W[0, 0:Theta_1_size]
    Theta_2_flat =  W[0, Theta_1_size:(Theta_1_size + Theta_2_size)]

    # reshape W into Thetas
    Theta_1 = Theta_1_flat.reshape(hidden_layer, L_in + 1)
    Theta_2 = Theta_2_flat.reshape(L_out, hidden_layer + 1)

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

    # compute the cost
    J = np.sum((-y.T.dot(np.log(h)) - (1 - y.T).dot(np.log(1 - h))) / m)

    # add regularization term
    # set ones column to zero
    temp_Theta_1 = Theta_1.copy()
    temp_Theta_2 = Theta_2.copy()
    temp_Theta_1[:, 0] = 0
    temp_Theta_2[:, 0] = 0

    # square thetas and multiply by lambda term
    squared_Thetas = np.sum(np.square(temp_Theta_1)) + np.sum(np.square(temp_Theta_2))
    reg_term = (L / (2 * m)) * squared_Thetas

    # add reg term to J
    J += reg_term

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

    return J, grad

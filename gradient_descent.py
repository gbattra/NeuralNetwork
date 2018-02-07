# runs gradient descent utilizing backpropagation to determine gradients

import numpy as np
from sigmoid import sigmoid
from compute_cost import compute_cost


def gradient_descent(W, L_in, hidden_layer, L_out, X, y, L, alpha, num_iters):
    J_history = np.zeros(num_iters)
    for i in range(0, num_iters):
        J, grad, h = compute_cost(W, L_in, hidden_layer, L_out, X, y, L)
        W -= alpha * grad

        # add J to J history
        J_history[i] = J

    return W, h, J_history
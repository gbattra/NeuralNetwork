# compute the cost for the neural network
# returns J and grad for a given Theta set

import numpy as np


def compute_cost(W, L_in, hidden_layer, L_out, X, y, L):

    # unwrap W into initial thetas
    Theta_1_size = (hidden_layer * (L_in + 1))
    Theta_2_size = (L_out * (hidden_layer + 1))
    Theta_1_flat = W[0, 0:Theta_1_size]
    Theta_2_flat =  W[0, Theta_1_size:(Theta_1_size + Theta_2_size)]

    # reshape W into Thetas
    Theta_1 = Theta_1_flat.reshape(hidden_layer, L_in + 1)
    Theta_2 = Theta_2_flat.reshape(L_out, hidden_layer + 1)

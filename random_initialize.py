# randomly initialize params for nn

import numpy as np
from epsilon_init import epsilon_init


def initialize_weights(L_in, L_out):

    # initialize Theta as zeros matrix
    W = np.zeros([L_out, L_in + 1])

    # initialize epsilon <-- the range of random values
    e_init = epsilon_init(L_in, L_out)

    # initialize random weights
    W = np.random.rand(W.shape[0], W.shape[1]).dot(2).dot(e_init).dot(e_init)

    return W

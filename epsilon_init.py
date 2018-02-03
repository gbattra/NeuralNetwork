# initialize epsilon based on nn features

import numpy as np


def epsilon_init(L_in, L_out):
    e_init = (np.sqrt(6)) / (np.square((L_in + L_out)))

    return e_init

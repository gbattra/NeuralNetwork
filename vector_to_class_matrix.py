# converts a y vector into a y matrix where the 1 column index is the class for this training example
import numpy as np


def vector_to_class_matrix(y, num_labels):

    # get training size
    m = y.shape[0]

    # initialize new placeholder y
    y_reshape = np.zeros((m, num_labels))
    for i in range(0, m):
        new_y = np.zeros((1, num_labels))
        new_y[0, np.int(y[i] - 1)] = 1  # minus 1 to y as array is 0 indexed
        y_reshape[i] = new_y

    return y_reshape

# scale features for better nn training

import numpy as np


def feature_scale(X):

    # get the mean of the features
    X_mean = np.mean(X)

    # subtract mean from X and divide by standard deviation
    X_norm = (X - X_mean) / np.std(X)

    return X_norm

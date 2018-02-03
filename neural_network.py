# this is a neural network model that predicts presence and type of arrhythmia in patients using 280 attributes

import numpy as np
import pandas as pd
import matplotlib as plt
import random_initialize as ri
import datacleaner
import scipy
from feature_scale import feature_scale
from compute_cost import compute_cost

# load and clean data
data = pd.read_csv('dataset.csv')
clean_data = datacleaner.autoclean(data, True).values
X = clean_data[:, 0:279]
y = clean_data[:, 279:280]

# perform feature scaling
X = feature_scale(X)

# initialize neural network features
input_layer = X.shape[1]  # number of initial attributes
hidden_layer = 50  # size of hidden layer of nn
num_labels = 16  # number of classes to predict
L = 1 # initialize Lambda

# randomly initialize Thetas
Theta_1 = ri.initialize_weights(input_layer, hidden_layer)
Theta_2 = ri.initialize_weights(hidden_layer, num_labels)

# roll up these Thetas so that they be used in the optimization function
W = np.matrix(np.append(Theta_1.flatten(), Theta_2.flatten()))

# test cost function for passing to optimization function
J, grad = compute_cost(W, input_layer, hidden_layer, num_labels, X, y, L)

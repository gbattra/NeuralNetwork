# this is a neural network model that predicts presence and type of arrhythmia in patients using 280 attributes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random_initialize as ri
import datacleaner
import scipy
from scipy import optimize
from sklearn import preprocessing
from compute_cost import compute_cost
from gradient_descent import gradient_descent
from encode_y import encode_y

# load and clean data
data = pd.read_csv('dataset.csv')
clean_data = datacleaner.autoclean(data, True).values
X = clean_data[:, 0:279]
y = clean_data[:, 279:280]

# perform feature scaling
X = preprocessing.scale(X)

# initialize neural network features
input_layer = X.shape[1]  # number of initial attributes
hidden_layer = 50  # size of hidden layer of nn
num_labels = 16  # number of classes to predict
L = 1  # initialize Lambda

# randomly initialize Thetas
Theta_1 = ri.initialize_weights(input_layer, hidden_layer)
Theta_2 = ri.initialize_weights(hidden_layer, num_labels)

# roll up these Thetas so that they be used in the optimization function
W = np.hstack((Theta_1.flatten(), Theta_2.flatten()))

# one hot encode y with my own garbage encoder
# where the column index for the value 1 indicates the corresponding class
y_reshape = encode_y(y, num_labels)

# test cost function for passing to optimization function
J, grad, h = compute_cost(W, input_layer, hidden_layer, num_labels, X, y_reshape, L)

# initialize training params
alpha = 0.001
num_iters = 1500

# train Thetas using gradient descent
W, h, J_history = gradient_descent(W, input_layer, hidden_layer, num_labels, X, y_reshape, L, alpha, num_iters)

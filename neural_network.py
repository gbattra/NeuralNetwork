# this is a neural network model that predicts presence and type of arrhythmia in patients using 280 attributes

import numpy as np
import pandas as pd
import scipy
import matplotlib as plt
import datacleaner
from feature_scale import feature_scale
import random_initialize as ri

# load and clean data
data = pd.read_csv('dataset.csv')
clean_data = datacleaner.autoclean(data, True).values
X = clean_data[:, 0:279]
y = clean_data[:, 279:280]

# perform feature scaling
X = feature_scale(X)

# initialize neural network features
input_layer_size = X.shape[1]  # number of initial attributes
hidden_layer_size = 50  # size of hidden layer of nn
num_labels = 16  # number of classes to predict

# randomly initialize Thetas
Theta_1 = ri.initialize_weights(input_layer_size, hidden_layer_size)
Theta_2 = ri.initialize_weights(hidden_layer_size, num_labels)

# this is a neural network model that predicts presence and type of arrhythmia in patients using 280 attributes

import numpy as np
import matplotlib.pyplot as plt
import random_initialize as ri
from sklearn import preprocessing
from sklearn import datasets
from compute_cost import compute_cost, get_cost, get_grad
from gradient_descent import gradient_descent
from predict import predict
from theta_transform import unwrap_theta
from scipy import optimize

# load example dataset from sklearn
data = datasets.load_breast_cancer()
X = data.data
y = data.target

# PREPROCESSING
# feature scale X
scaler = preprocessing.StandardScaler()
scaler.fit(X)
X = scaler.fit_transform(X)

# get features dimensions
m, n = X.shape

# one hot encode y
onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
y_encoded = onehot_encoder.fit_transform(np.matrix(y).T)

# initialize neural network features
input_layer = X.shape[1]  # number of initial attributes
hidden_layer = 15  # size of hidden layer of nn
num_labels = y_encoded.shape[1]  # number of classes to predict
L = 1  # initialize Lambda

# randomly initialize Thetas
Theta_1 = ri.initialize_weights(input_layer, hidden_layer)
Theta_2 = ri.initialize_weights(hidden_layer, num_labels)

# roll up these Thetas so that they be used in the optimization function
W = np.hstack((Theta_1.flatten(), Theta_2.flatten()))

# test cost function for passing to optimization function
J, grad = compute_cost(W, input_layer, hidden_layer, num_labels, X, y_encoded, L)

# initialize training params
alpha = 1.0
num_iters = 1500

# train Thetas using gradient descent
W, J_history = gradient_descent(W, input_layer, hidden_layer, num_labels, X, y_encoded, L, alpha, num_iters)

# plot J_history to make sure gradient descent worked properly
plt.plot(range(num_iters), J_history)

# unwrap trained thetas
Theta_1, Theta_2 = unwrap_theta(W, input_layer, hidden_layer, num_labels)

# make final predictions and compute accuracy
h = predict(X, [Theta_1, Theta_2])
predicted_classes = h.argmax(axis=1)
err = np.count_nonzero(predicted_classes - y)
gd_acc = (m - err) / float(m)
gd_cost, gd_grad = compute_cost(W, input_layer, hidden_layer, num_labels, X, y_encoded, L)

# now with scipy's fmin_cg function to see how gradient descent compared
W = optimize.fmin_cg(get_cost, W, args=(input_layer, hidden_layer, num_labels, X, y_encoded, L))
Theta_1, Theta_2 = unwrap_theta(W, input_layer, hidden_layer, num_labels)
h = predict(X, [Theta_1, Theta_2])
predicted_classes = h.argmax(axis=1)
err = np.count_nonzero(predicted_classes - y)
fmincg_acc = (m - err) / float(m)

# print gradient descent accuracy (~ 98%)
print('GD Cost: ' + str(gd_cost) + '\n')
print('GD Accuracy: ' + str(gd_acc) + '\n')

# print fmin_cg accuracy (~ 98%)
fmincg_cost, fmincg_grad = compute_cost(W, input_layer, hidden_layer, num_labels, X, y_encoded, L)
print('fmin_cg Cost: ' + str(fmincg_cost) + '\n')
print('fmin_cg Accuracy: ' + str(fmincg_acc) + '\n')

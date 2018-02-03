# this is a neural network model that predicts presence and type of arrhythmia in patients using 280 attributes

import numpy as np
import pandas as pd
import scipy
import matplotlib as plt
import datacleaner
from feature_scale import feature_scale

# load and clean data
data = pd.read_csv('dataset.csv')
clean_data = datacleaner.autoclean(data, True).values
X = clean_data[:, 0:279]
y = clean_data[:, 279:280]

# perform feature scaling
X = feature_scale(X)


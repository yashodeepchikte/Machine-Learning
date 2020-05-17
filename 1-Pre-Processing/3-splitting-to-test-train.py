# Splitting to test and train\

# make sore to split before feature scaling
# as the train data must not be feature scaled with the training data
# either train or test must not have knowlwdge of each other

#  <---------------- this part is common  ------------------->
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('Data.csv')       # ----> Set the correct path
X = dataset.iloc[:, :-1].values         # ----> Set the correct columns
y = dataset.iloc[:, 3].values           # ----> set the correct rows
#  <---------------- this part is common  ------------------->

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

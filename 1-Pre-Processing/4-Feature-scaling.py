# feature scaling
# make sure all categorical data has been delt with

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

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))

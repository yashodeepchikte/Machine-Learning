#missing data

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


# Taking care of missing data
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
missingvalues = missingvalues.fit(X[:, 1:3])
X[:, 1:3]=missingvalues.transform(X[:, 1:3])

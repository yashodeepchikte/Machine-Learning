# Handelling the categorical encodeoing

# make sure all the missing values and NaN have been taken care of

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


# Encoding categorical data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Encoding the Independent Variable
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough') # ---> correct columns
X = np.array(ct.fit_transform(X), dtype=np.float)

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)        # ---> do this only when y id categorical

#missing data

#  <---------------- Importing data  ------------------->
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('Data.csv')       # ----> Set the correct path
X = dataset.iloc[:, :-1].values         # ----> Set the correct columns
y = dataset.iloc[:, 3].values           # ----> set the correct rows



# <------------- Taking care of missing data -------------------->
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
missingvalues = missingvalues.fit(X[:, 1:3])
X[:, 1:3]=missingvalues.transform(X[:, 1:3])

# <-------------- Encoding categorical data ------------------------->
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Encoding the Independent Variable
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough') # ---> correct columns
X = np.array(ct.fit_transform(X), dtype=np.float)

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)        # ---> do this only when y id categorical

# <----------------------- Splitting to test and train --------------->
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# <------------------------- Feature Scaling ------------------------->
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))

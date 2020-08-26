# Multiple linear Regerssion

#  <---------------- Importing data  ------------------->
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')       # ----> Set the correct path
X = dataset.iloc[:, :-1].values         # ----> Set the correct columns
y = dataset.iloc[:, -1].values           # ----> set the correct rows



# <------------- Taking care of missing data -------------------->
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
missingvalues = missingvalues.fit(X[:, 0:3])
X[:, 0:3]=missingvalues.transform(X[:, 0:3])

# <-------------- Encoding categorical data ------------------------->
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Encoding the Independent Variable
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough') # ---> correct columns
X = np.array(ct.fit_transform(X), dtype=np.float)
X = X[:, 1:]

# Encoding the Dependent Variable
#labelencoder_y = LabelEncoder()
#y = labelencoder_y.fit_transform(y)        # ---> do this only when y id categorical

# <----------------------- Splitting to test and train --------------->
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# <------------------------- Feature Scaling ------------------------->
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# < ------------------- Multiple linear regression ------------------------>
# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# making predictions
y_pred = regressor.predict(X_test)

plt.scatter(y_test, y_pred)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.title("y_pted vs y_test")

m,b = np.polyfit(y_test, y_pred, deg=1)
plt.plot(y_test, m*y_test+b)

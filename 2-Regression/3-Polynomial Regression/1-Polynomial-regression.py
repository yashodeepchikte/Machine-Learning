# Preprocessiong template

#  <---------------- Importing data  ------------------->
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')       # ----> Set the correct path
X = dataset.iloc[:, 1].values         # ----> Set the correct columns
X = X.reshape(-1,1)
y = dataset.iloc[:, -1].values           # ----> set the correct rows



# <------------- Taking care of missing data -------------------->
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
missingvalues = missingvalues.fit(X[:])
X[:]=missingvalues.transform(X[:])

# <-------------- Encoding categorical data ------------------------->
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.compose import ColumnTransformer
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Encoding the Independent Variable
#ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough') # ---> correct columns
#X = np.array(ct.fit_transform(X), dtype=np.float)

# Encoding the Dependent Variable
#labelencoder_y = LabelEncoder()
#y = labelencoder_y.fit_transform(y)        # ---> do this only when y id categorical

# <----------------------- Splitting to test and train --------------->
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# <------------------------- Feature Scaling ------------------------->
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)

# <-----------------------fitting polynomial regression --------------------->
# we will map X to a specified degree then use linear regression
# Fitting Polynomial Regression to the dataset

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)          # --> poly_reg.fit_transform will return an array of all comb^n of powers
poly_reg.fit(X_poly, y)
# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

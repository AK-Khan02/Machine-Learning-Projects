# Import packages
import pandas as pd

# Load dataset
col_names = ['income', 'age', 'num_rooms', 'num_bedrooms', 'population', 'price', 'address']
housing = pd.read_csv('Housing_Data.csv', header=1, names=col_names)

# Inspect dataset
housing.head()

import seaborn as sns
sns.pairplot(housing) # This line takes a while to run

"""Most of the variables appear to be normally distributed, and seem to be positively correlated with price."""

# Split dataset into features and target variable
feature_cols = ['income', 'age', 'num_rooms', 'num_bedrooms', 'population']
X = housing[feature_cols]
y = housing.price

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=546) # Set random_state for reproducibility

from sklearn.linear_model import LinearRegression

# Create the model
linreg = LinearRegression()

# Fit the model using training data
linreg.fit(X_train, y_train)

print(linreg.coef_)
print(linreg.intercept_)

pd.DataFrame(linreg.coef_, X.columns, columns = ['Coefficient'])

pred = linreg.predict(X_test)

import matplotlib.pyplot as plt
plt.scatter(y_test, pred)

plt.hist(y_test - pred)

from sklearn import metrics
import numpy as np

print(metrics.mean_absolute_error(y_test, pred)) # MAE
print(metrics.mean_squared_error(y_test, pred))  # MSE
print(np.sqrt(metrics.mean_squared_error(y_test, pred))) # There's no method for RMSE, so we calculate it manually

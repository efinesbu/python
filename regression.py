import pandas as pd
import numpy as np
import pylab
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import math

# Increase default column visibility
desired_width = 500
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 26)
pd.set_option('display.max_rows', 100)

# Import data with comma separated values and python engine
auto_data = pd.read_csv('../data/imports-85.data', sep=r'\s*,\s*', engine='python')

# Refine column names, they were missing from the data file for some reason
auto_data.columns = ['symboling','normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 'cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']

## Replace ? values with NaN
auto_data = auto_data.replace('?', np.nan)

## View various stats on your data
#print(auto_data.describe(include='all'))

# # View details on single column | Object types cannot be summerized in .describe() with stats, needs to be converted to a float
# Convert Price to Float
auto_data['price'] = pd.to_numeric(auto_data['price'], errors='coerce')

# Check column type after update
#print(auto_data['price'].describe())

# Drop data columns that do not help predict the price of a vehicle
auto_data = auto_data.drop('normalized-losses', axis=1)

# Convert Horsepower from Object to Float
auto_data['horsepower'] = pd.to_numeric(auto_data['horsepower'], errors='coerce')

#Assign numeric values to text that make sense for Cylinders
cylinders_dict={'two':2,
                'three':3,
                'four':4,
                'five':5,
                'six':6,
                'eight':8,
                'twelve':12}
auto_data['cylinders'].replace(cylinders_dict, inplace=True)

# Convert multiple categorical values into numerical sets using one-hot method
auto_data = pd.get_dummies(auto_data, columns=['make',
                                               'fuel-type',
                                               'aspiration',
                                               'num-of-doors',
                                               'body-style',
                                               'drive-wheels',
                                               'engine-location',
                                               'engine-type',
                                               'fuel-system'])

# Drop any rows with non-number values
auto_data = auto_data.dropna()

## Check if data contains any Null values
#print(auto_data[auto_data.isnull().any(axis=1)])

##################################################################################
# Feed data into ML model for training and testing

# Data set is everything besides price column
X = auto_data.drop('price', axis=1)

# Taking the labels (price) to be the Y axis
Y = auto_data['price']

# Splitting into 80% training set and 20% for testing set so we can see our accuracy
X_train, x_test, Y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=0)

# LinearRegression is our Estimator Object, an API to learn from data and apply fit() method
linear_model = LinearRegression()
linear_model.fit(X_train, Y_train)
#print(linear_model.fit(X_train, Y_train))

# Regression line R squared quality check
linear_model.score(X_train, Y_train)
#print(linear_model.score(X_train, Y_train))

# Retreive column names of data set
predictors = X_train.columns

# Create series of coefficients to feature names, and sort by coefficients | tells us the weight of a given feature
coef = pd.Series(linear_model.coef_,predictors).sort_values()
#print(coef)

##################################################################################
# Testing with Linear Regression
y_predict = linear_model.predict(x_test)

# pylab.rcParams['figure.figsize'] = (15, 6)
# plt.plot(y_predict, label='Predicted')
# plt.plot(y_test.values, label='Actual')
# plt.ylabel('Price')
#
# plt.legend()
# plt.show()

# # Regression quality test on test values
# r_square = linear_model.score(x_test, y_test)
# print("Linear Model Regression")
# print('R square:', r_square)
#
# # Mean Square Error
# linear_model_mse = mean_squared_error(y_predict, y_test)
# print(linear_model_mse)
#
# # RMSE - s.d of risiduals | avg expected error of prediction from actual
# print("RMSE: ", math.sqrt(ridge_model_mse))


##################################################################################
# Testing with Lasso Regression

# Set alpha to prevent overfitting | normalize data to center around 0
lasso_model = Lasso(alpha=5, normalize=True)
lasso_model.fit(X_train, Y_train)

coef = pd.Series(lasso_model.coef_, predictors).sort_values()

y_predict = lasso_model.predict(x_test)
# pylab.rcParams['figure.figsize'] = (15, 6)
# plt.plot(y_predict, label='Predicted')
# plt.plot(y_test.values, label='Actual')
# plt.ylabel('Price')
#
# plt.legend()
# plt.show()
#
# r_square = lasso_model.score(x_test,y_test)
# print("Lasso Model Regression")
# print('R square:', r_square)
#
# # # Mean Square Error
# lasso_model_mse = mean_squared_error(y_predict, y_test)
# print("RMSE: ", math.sqrt(ridge_model_mse))


##################################################################################
# Testing with Ridge Regression

ridge_model = Ridge(alpha=0.5, normalize=True)
ridge_model.fit(X_train, Y_train)
coef = pd.Series(ridge_model.coef_, predictors).sort_values()
y_predict = ridge_model.predict(x_test)
pylab.rcParams['figure.figsize'] = (15, 6)
plt.plot(y_predict, label='Predicted')
plt.plot(y_test.values, label='Actual')
plt.ylabel('Price')

plt.legend()
plt.show()
r_square = ridge_model.score(x_test,y_test)
print("Ridge Regression")
print('R square:', r_square)

# # Mean Square Error
ridge_model_mse = mean_squared_error(y_predict, y_test)
print("RMSE: ", math.sqrt(ridge_model_mse))

#Hyperparameter tuning determines the best alpha coefficients for your model
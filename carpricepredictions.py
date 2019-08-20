import pylab
import numpy as np
import pandas as pd
from pandas import Series
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import math

##################################################################################
# Increase default column visibility
desired_width = 500
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 26)
pd.set_option('display.max_rows', 100)

##################################################################################
# PREPROCESSING

# Import data with comma separated values and python engine
auto_data = pd.read_csv('data/imports-85.data', sep=r'\s*,\s*', engine='python', header=None)

# Refine column names, they were missing from the data file for some reason
auto_data.columns = ['symboling','normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 'cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']

# Replace ? values with NaN
auto_data = auto_data.replace('?', np.nan)

##View various stats on your data
#print(auto_data.describe(include='all'))

# View details on single column | Object types cannot be summerized in .describe() with stats, needs to be converted to a float
# Convert Price to Float
auto_data['price'] = pd.to_numeric(auto_data['price'], errors='coerce')

# Check column type after update
#print(auto_data['price'].describe())

# Drop data columns that do not help predict the price of a vehicle
auto_data = auto_data.drop('normalized-losses', axis=1)

# Convert Horsepower from Object to Float
auto_data['horsepower'] = pd.to_numeric(auto_data['horsepower'], errors='coerce')

# Assign numeric values to text that make sense for Cylinders
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

# Data set is composed of all columns besides price column
X = auto_data.drop('price', axis=1)

# Taking the labels (price) to be the Y axis
Y = auto_data['price']

# Splitting into 80% training set and 20% for testing set so we can see our accuracy
X_train, x_test, Y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=0)

##################################################################################
# LinearRegression is our Estimator Object, an API to learn from data and apply fit() method
linear_model = LinearRegression()
linear_model.fit(X_train, Y_train)
# Testing with Linear Regression
y_predict_linear = linear_model.predict(x_test)

##################################################################################
# Retreive column names of data set
predictors = X_train.columns

# Create series of coefficients to feature names, and sort by coefficients | tells us the weight of a given feature
coef = pd.Series(linear_model.coef_,predictors).sort_values()
#print(coef)

##################################################################################
# # Regression quality test on test values
r_square = linear_model.score(x_test, y_test)

# # Mean Square Error
linear_model_mse = mean_squared_error(y_predict_linear, y_test)
print('Linear Model| R square:', '{:.2%}'.format(r_square), '| RMSE : $', '{:.6}'.format(math.sqrt(linear_model_mse)))

##################################################################################
# HYPER TUNING - USE GRIDSEARCHCV TO FIND BEST PARAMETERS FOR LASSO MODEL
alpha_est = [3, 5, 9, 10]

param_grid = {'alpha': alpha_est}

grid_search = GridSearchCV(Lasso(normalize=True),
                           param_grid, cv=3, return_train_score=True)
grid_search.fit(X_train, Y_train)
print()
print('Best Lasso Alpha', grid_search.best_params_)

##################################################################################
# Testing with Lasso Regression

# Set alpha to prevent overfitting | normalize data to center around 0
lasso_model = Lasso(alpha=grid_search.best_params_['alpha'], normalize=True)
lasso_model.fit(X_train, Y_train)

coef = pd.Series(lasso_model.coef_, predictors).sort_values()

y_predict_lasso = lasso_model.predict(x_test)

r_square = lasso_model.score(x_test,y_test)
# # # Mean Square Error
lasso_model_mse = mean_squared_error(y_predict_lasso, y_test)
print('Lasso Model | R square:', '{:.2%}'.format(r_square), '| RMSE : $', '{:.6}'.format(math.sqrt(lasso_model_mse)))

##################################################################################
# HYPER TUNING - USE GRIDSEARCHCV TO FIND BEST PARAMETERS FOR RIDGE MODEL
alpha_est = [0.1, 0.5, 0.9]

param_grid = {'alpha': alpha_est}

grid_search = GridSearchCV(Ridge(normalize=True),
                           param_grid, cv=3, return_train_score=True)
grid_search.fit(X_train, Y_train)
print()
print('Best Ridge Alpha', grid_search.best_params_)

##################################################################################
# Testing with Ridge Regression
ridge_model = Ridge(alpha=grid_search.best_params_['alpha'], normalize=True)
ridge_model.fit(X_train, Y_train)
coef = pd.Series(ridge_model.coef_, predictors).sort_values()

y_predict_ridge = ridge_model.predict(x_test)

# # Mean Square Error
r_square = ridge_model.score(x_test,y_test)
ridge_model_mse = mean_squared_error(y_predict_ridge, y_test)
print('Ridge Model | R square:', '{:.2%}'.format(r_square), '| RMSE : $', '{:.6}'.format(math.sqrt(ridge_model_mse)))


##################################################################################
# HYPER TUNING - USE GRIDSEARCHCV TO FIND BEST PARAMETERS FOR GRADIENT BOOSTING MODEL

num_estimators = [100, 200, 500]
learn_rates = [0.01, 0.02, 0.05, 0.1]
max_depths = [4, 6, 8]

param_grid = {'n_estimators': num_estimators,
              'learning_rate': learn_rates,
              'max_depth': max_depths}

grid_search = GridSearchCV(GradientBoostingRegressor(min_samples_split=2, loss='ls'),
                           param_grid, cv=3, return_train_score=True)
grid_search.fit(X_train, Y_train)
print()
print('Best Gradient Boosting Parameters: ', grid_search.best_params_)

##################################################################################
# Implement Gradient Boosting Regression

# Hardcoded for speed
# params = {'n_estimators': grid_search.best_params_['n_estimators'], 'max_depth': grid_search.best_params_['max_depth'], 'min_samples_split': 2, 'learning_rate': grid_search.best_params_['learning_rate'], 'loss': 'ls'}
params = {'n_estimators': 200, 'max_depth': 4, 'min_samples_split': 2, 'learning_rate': 0.05, 'loss': 'ls'}
gbr_model = GradientBoostingRegressor(**params)
gbr_model.fit(X_train, Y_train)

##################################################################################
# Make predictions using the trained Gradient Boosting Model & test data
y_predict_gbr = gbr_model.predict(x_test)
gbr_model_mse = mean_squared_error(y_predict_gbr, y_test)
r_square = gbr_model.score(x_test, y_test)
print('GB Model    | R square:', '{:.2%}'.format(r_square), '| RMSE : $', '{:.6}'.format(math.sqrt(gbr_model_mse)))

##################################################################################
c_est = [.01, .05, .1, .5, .7, 1]

param_grid = {'C': c_est}

grid_search = GridSearchCV(SVR(kernel='linear'),
                           param_grid, cv=3, return_train_score=True)
grid_search.fit(X_train, Y_train)
print()
print('Best SVM Penalty Factor ', grid_search.best_params_)

##################################################################################
# SUPPORT VECTOR MODEL| c = penalty factor
regression_model = SVR(kernel='linear', C=grid_search.best_params_['C'])
regression_model.fit(X_train, Y_train)

##################################################################################
# Plot coefficient impact on price
predictors = X_train.columns
coef = Series(regression_model.coef_[0], predictors).sort_values()
#coef.plot(kind='bar', title='Modal Coefficients')

##################################################################################
# RUN PREDICTION ON TEST DATA
y_predict_svm = regression_model.predict(x_test)

##################################################################################
# QUALITY CHECK
r_square = regression_model.score(x_test, y_test)
regression_model_mse = mean_squared_error(y_predict_svm, y_test)
print('SVM Model   | R square:', '{:.2%}'.format(r_square), '| RMSE : $', '{:.6}'.format(math.sqrt(regression_model_mse)))

##################################################################################
pylab.rcParams['figure.figsize'] = (15, 6)
plt.plot(y_predict_linear, label='Predicted Linear')
plt.plot(y_predict_lasso, label='Predicted Lasso')
plt.plot(y_predict_ridge, label='Predicted Ridge')
plt.plot(y_predict_gbr, label='Predicted GB')
plt.plot(y_predict_svm, label='Predicted SVM')
plt.plot(y_test.values, label='Actual Price')
plt.ylabel('Price')
plt.legend()
plt.show()

import pandas as pd
from pandas import Series
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import math
import pylab
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

desired_width = 500
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 26)
pd.set_option('display.max_rows', 100)

auto_data = pd.read_csv('../demo/auto-mpg.data', delim_whitespace=True, header=None,
                        names = ['mpg',
                                 'cylinders',
                                 'displacement',
                                 'horepower',
                                 'weight',
                                 'acceleration',
                                 'model',
                                 'origin',
                                 'car_name'])

# Drop Car Names since they're mostly unique and dont impact mpg
auto_data = auto_data.drop('car_name', axis=1)

# Define origin mappings and then one-hot vector them
auto_data['origin'] = auto_data['origin'].replace({1:'america', 2:'europe',3:'asia'})
auto_data = pd.get_dummies(auto_data, columns=['origin'])

# Replace ? with NaN
auto_data = auto_data.replace('?', np.nan)

# Drop non-numeric rows
auto_data = auto_data.dropna()

# Train
X = auto_data.drop('mpg', axis=1)

Y = auto_data['mpg']

X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# c = penalty factor
regression_model = SVR(kernel='linear', C=0.5)
regression_model.fit(X_train, Y_train)

# Plot coefficient impact on mpg
predictors = X_train.columns
coef = Series(regression_model.coef_[0], predictors).sort_values()
#coef.plot(kind='bar', title='Modal Coefficients')

y_predict = regression_model.predict(x_test)

pylab.rcParams['figure.figsize'] = (15,6)
plt.plot(y_predict, label='Predicted')
plt.plot(y_test.values, label='Actual')
plt.ylabel('MPG')

plt.legend()
plt.show()

regression_model_mse = mean_squared_error(y_predict, y_test)
print("RSME", math.sqrt(regression_model_mse))

#print(regression_model_mse)
#print(regression_model.score(x_test, y_test))


#print(auto_data.head())
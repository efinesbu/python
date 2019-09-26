# Copyright(C) Emil Fine
# This program demonstrates principal component analysis (PCA) on wine and uses reduced dimensionality
# reduction data set to predict quality of wine
print('''
This program returns the probability of guessing wine quality from 7 categories using Support Vector Machines 
with and without Principle Component Analysis & Dimension Reduction. 
''')

##################################################################################
# IMPORTS
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

##################################################################################
# INCREASE DEFAULT COLUMN VISIBILITY

desired_width = 500
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 26)
pd.set_option('display.max_rows', 100)

##################################################################################
# IMPORT DATA | NO CLEANING REQUIRED
wine_data = pd.read_csv('data/winequality-white.csv',
                        names=['Fixed Acidity',
                               'Volatile Acidity',
                               'Citric Acid',
                               'Residual Sugar',
                               'Chlorides',
                               'Free Sulfur Dioxide',
                               'Total Sulfur Dioxide',
                               'Density',
                               'pH',
                               'Sulphates',
                               'Alcohol',
                               'Quality'
                               ],
                        skiprows=1, # SKIP HEADER ROW
                        sep=r'\s*;\s*', # SEPARATOR IS A SEMICOLON
                        engine='python')

##################################################################################
# DETERMINE HOW MANY DIFFERENT QUALITY OPTIONS THERE ARE | 7 so a random guess is (1/7) 14% accurate
print('Method 1: Is a random guess (1/7): ' + "{:.2%}".format(1/len(wine_data['Quality'].unique())))
# {:.2%}".format(x)

##################################################################################
# PCA ONLY USES X VARIABLE AXIS FOR UNSUPERVIZED LEARNING TECHNIQUE
x = wine_data.drop('Quality', axis=1)
y = wine_data['Quality']

# STANDARDIZE X DATA BY SUBTRACTING THE MEAN AND DIVIDING BY S.D. (PREPROCESSING)
x = preprocessing.scale(x)

# SPLIT DATA
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

##################################################################################
# SVM CLASSIFIER USING ALL FEATURES OF INPUT DATA

# LinearSVC()
clf_cvs = LinearSVC(penalty='l1', dual=False, tol=1e-3)
clf_cvs.fit(x_train, y_train)

##################################################################################
# PERFORM PREDICTIONS ON TEST DATA
m2_accuracy = clf_cvs.score(x_test, y_test)
print('Method 2: R-score using SVM with 11 dimensions: '+ "{:.2%}".format(m2_accuracy))

##################################################################################
# SHOW HEATMAP
corrmat = wine_data.corr()
f, ax = plt.subplots(figsize=(7, 7))
sns.set(font_scale=.8)
sns.heatmap(corrmat, square=True, annot=True, fmt='.2f', cmap='winter').set_ylim(12.0, 0)
plt.show()

##################################################################################
# PERFORM PCA
component_count = 11 #SET NUMBER OF COMPONENTS
pca = PCA(n_components=component_count, whiten=True)
X_reduced = pca.fit_transform(x) # CALL FIT TRANSFORM TO RUN PCA ON DATA SET
constant = pca.explained_variance_ratio_ # MAGNITUDE OF VARIATION BY COMPONENTS

##################################################################################
# DETERMINE HOW MANY DIMENSIONS YOU NEED | PLOT DIMENSIONS | SCREE PLOT - FIND ELBOW

plt.plot(constant)
plt.xlabel("Dimensions")
plt.ylabel('Explain Variance Ratio')
plt.show()

##################################################################################
# SET NEW DIMENSION COUNT AFTER PCA | DISCARD DIMENSIONS/COMPONENTS WITH LOW IMPACT
component_count_new = 1 # SET NUMBER OF COMPONENTS | EDIT DIMENSION COUNT HERE TO SEE IMPACT TO ACCURACY
pca = PCA(n_components=component_count_new, whiten=True)
X_reduced = pca.fit_transform(x) # CALL FIT TRANSFORM TO RUN PCA ON DATA SET

##################################################################################
# RETRAIN WITH DECOMPOSED DATA
x_train, x_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=0)
clf_cvs = LinearSVC(penalty='l1', dual=False, tol=1e-3)
clf_cvs.fit(x_train, y_train)
m3_accuracy = clf_cvs.score(x_test, y_test)
print('Method 3: R-score using SVM with',
      component_count_new,
      'dimension(s): ' + "{:.2%}".format(m3_accuracy)) #NEED TO REDUCE COMPONENTS TO NOTICE DIFFERENCE
print("Result is a",
      abs(component_count-component_count_new),
      "x change in speed and a " + "{:.2%}".format(m3_accuracy - m2_accuracy),
      "change in accuracy")
print()
print("Note: Change line 94 variable: 'component_count_new' to see impact of using different dimension amounts")


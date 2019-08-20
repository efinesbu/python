# This program clusters data of titanic passengers into most meaningful groups

##################################################################################
import pandas as pd
##################################################################################
# Increase default column visibility
desired_width = 500
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 26)
pd.set_option('display.max_rows', 100)

##################################################################################
# TAKE IN PASSANGER DATA
titanic_data = pd.read_csv('data/train.csv', quotechar='"')

##################################################################################
# DROP DATA THAT IS TOO SPECIFIC
titanic_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], 'columns', inplace=True)

##################################################################################
# CONVERT GENDER INTO NUMERICAL FORM | preprocessing.labelencoder for binary data
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
titanic_data['Sex'] = le.fit_transform(titanic_data['Sex'].astype(str))

##################################################################################
# CONVERT EMBARKED CATEGORICAL DATA INTO NUMERIC WITH ONE-HOT
titanic_data = pd.get_dummies(titanic_data, columns=['Embarked'])

##################################################################################
## CHECK FOR INVALID DATA (OPTIONAL)
#print(titanic_data[titanic_data.isnull().any(axis=1)])

##################################################################################
# DROP INVALID DATA
titanic_data = titanic_data.dropna()

##################################################################################
# DETERMINE GOOD BANDWIDTH BASED ON DATA
# SEPARATED FOR SPEED
# OPTIONAL - AUTOMATICALLY USED IF NOT SPECIFIED

# from sklearn.cluster import estimate_bandwidth
# print(estimate_bandwidth(titanic_data))
# 30.44675914497196

##################################################################################
# IMPLEMENT MEANSHIFT ANALYSER
from sklearn.cluster import MeanShift
analyzer = MeanShift(bandwidth=30.44675914497196)

##################################################################################
# CALL FIT METHOD TO TRAIN DATA | ASSIGNS  PASSENGERS TO CLUSTERS
analyzer.fit(titanic_data)

##################################################################################
# HOW MANY CLUSTERS YOUR DATA IS SPLIT INTO BASED ON BANDWIDTH
import numpy as np
labels = analyzer.labels_
# print(np.unique(labels))

##################################################################################
# WHAT CLUSTER EACH PASSENGER IS PART OF
titanic_data['cluster_group'] = np.nan
data_length = len(titanic_data)
for i in range(data_length):
    titanic_data.iloc[i, titanic_data.columns.get_loc('cluster_group')] = labels[i]

##################################################################################
# GROUP PASSENGERS BY CLUSTER
titanic_cluster_data = titanic_data.groupby(['cluster_group']).mean()

##################################################################################
# IMPROVE DATA BY ADDING COUNT COLUMN
titanic_cluster_data['Counts'] = pd.Series(titanic_data.groupby(['cluster_group']).size())

#print(titanic_data[titanic_data['cluster_group']==1].describe())
titanic_cluster_data.drop(['SibSp', 'Parch'], 'columns', inplace=True)
print(titanic_cluster_data.head())

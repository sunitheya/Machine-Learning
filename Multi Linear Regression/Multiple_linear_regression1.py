#Multiple Linear Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing data

dataset = pd.read_csv('50_startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,4].values

# Encoding catagorical data
#Encoding the independent variable

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3]) # OneHotEncoder is to make values as categorical
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the dummy variable trap
X = X[:,1:]


#spliiting data into test and training
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)


#fitting MLR to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the test set results
y_pred = regressor.predict(X_test)
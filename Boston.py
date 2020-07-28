# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 18:19:26 2019

@author: Dipesh
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

trdata=pd.read_csv('train.csv')
tsddata=pd.read_csv('test.csv')
sub=pd.read_csv('submission_example.csv')

trdata['medv'].mean()
cor_matrix=trdata.corr()

trdata.chas.value_counts()

trdata.rad.value_counts()
trdata['age'].mean()


df = pd.DataFrame({'values':trdata['medv'].values,'rooms':trdata['rm'].values})
X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

value=trdata.iloc['medv'].values
rooms=trdata.iloc['rm'].values
value.reshape(1,-1)
rooms.reshape(1,-1)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X, y)

plt.scatter(X, y,color='red')
plt.plot(X,regressor.predict(X), color = 'blue')
plt.title('Boston Housing')
plt.xlabel('Value of house/$1000')
plt.ylabel('Rooms in house') 
plt.show()

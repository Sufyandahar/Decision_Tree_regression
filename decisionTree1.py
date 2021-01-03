#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 12:47:43 2020

@author: sufiyan
"""

# import dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read csv file 
dataset = pd.read_csv('Position_Salaries.csv')

# slicing the dataset
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

regressor.predict([[6.5]])

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()
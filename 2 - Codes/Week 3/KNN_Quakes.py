# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 14:48:06 2020
https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
@author: Ahmed_A_Torky
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler

# Assign colum names to the dataset
M_Range = ['4', '5', '6', '7', '8', '9']

# =============================================================================
# CLASSIFICATION
# =============================================================================
# Read dataset to pandas dataframe
dataset = pd.read_csv('quakes_values.csv')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
print(dataset.head(None))
dataset_values = dataset.values
# split into input (X) and output (y) variables
X = dataset[['NumberOfSites','PeakAcceleration(gal)','Intensity']].values
y = dataset_values[:,3]
# y = np.round(y * 2) / 2 
# y = np.round(y * 5) / 5 
# y = np.round(y * 4) / 4 
y = y.astype(int)
y = y.astype(str)
try:
    if y.shape[1]:  
        ny = y.shape[1]
except IndexError:
    ny = 1
    # y = np.reshape(y, [y.shape[0],1])
try:
    if X.shape[1]:  
        nX = X.shape[1]
except IndexError:
    nX = 1
    X = np.reshape(X, [X.shape[0],1])
# Scale X
scaler_X  = StandardScaler() 
scaler_X.fit(X)
X_scaled = scaler_X.transform(X)
# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.10) 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

error = []
# Calculating error for K values between 3 and 30
for i in range(1, 100):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
plt.figure(figsize=(12, 6))
plt.plot(range(1, 100), error, color='red', linestyle='dashed', marker='o',
          markerfacecolor='blue', markersize=4)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

# Optimum from Curve
ngh = 8
classifier = KNeighborsClassifier(n_neighbors=ngh)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=1))

Name = 'Compare with N Neighbors = '+ np.str(ngh)
y_test = y_test.astype(float)
y_pred = y_pred.astype(float)
# compare y_test (scatter plot)
nx = 20
q_x = np.linspace(1, nx, nx)
fig = plt.figure(figsize=(10,7))
fsS=12
ax = fig.add_subplot(1, 1, 1)
ax.set_title(Name,fontsize=fsS)
ax.scatter(q_x,y_test[:nx], marker= 'o', s=70, facecolors='none', edgecolors='tab:blue', label = 'actual')
ax.scatter(q_x,y_pred[:nx], marker= 'x', s=70, color='tab:orange', label = 'predicted')
ax.grid()
plt.ylabel('M')
plt.xlabel('quake')
plt.legend(loc='upper left')
plt.show()

# =============================================================================
# REGRESSION
# =============================================================================
# # Read dataset to pandas dataframe
# dataset = pd.read_csv('quakes_values.csv')
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
# print(dataset.head(None))
# dataset_values = dataset.values
# # split into input (X) and output (y) variables
# X = dataset[['NumberOfSites','PeakAcceleration(gal)','Intensity']].values
# y = dataset_values[:,3]
# # y = np.round(y * 2) / 2 
# # y = np.round(y * 4) / 4 
# # y = y.astype(int)
# # y = y.astype(str)
# try:
#     if y.shape[1]:  
#         ny = y.shape[1]
# except IndexError:
#     ny = 1
#     # y = np.reshape(y, [y.shape[0],1])
# try:
#     if X.shape[1]:  
#         nX = X.shape[1]
# except IndexError:
#     nX = 1
#     X = np.reshape(X, [X.shape[0],1])
# # Scale X
# scaler_X  = StandardScaler() 
# scaler_X.fit(X)
# X_scaled = scaler_X.transform(X)
# # Scale y
# scaler_y  = StandardScaler() 
# scaler_y.fit(y)
# y_scaled = scaler_y.transform(y)
# # split into train and test datasets
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.10) 
# from sklearn.neighbors import KNeighborsRegressor
# error = []
# y_test=scaler_y.inverse_transform(y_test)
# # Calculating error for K values between 3 and 30
# for i in range(1, 100):
#     knn = KNeighborsRegressor(n_neighbors=i)
#     knn.fit(X_train, y_train)
#     pred_i = knn.predict(X_test)
#     pred_i=scaler_y.inverse_transform(pred_i)
#     error.append(np.mean(np.abs(y_test-pred_i)*100 / y_test))
# plt.figure(figsize=(12, 6))
# plt.plot(range(1, 100), error, color='red', linestyle='dashed', marker='o',
#           markerfacecolor='blue', markersize=4)
# plt.title('Error Rate K Value')
# plt.xlabel('K Value')
# plt.ylabel('Mean Error')

# # Optimum from Curve
# ngh = 15
# classifier = KNeighborsRegressor(n_neighbors=ngh)
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)
# y_pred=scaler_y.inverse_transform(y_pred)
# Name = 'Compare with N Neighbors = '+ np.str(ngh)
# # compare y_test (scatter plot)
# nx = 20
# q_x = np.linspace(1, nx, nx)
# fig = plt.figure(figsize=(10,7))
# fsS=12
# ax = fig.add_subplot(1, 1, 1)
# ax.set_title(Name,fontsize=fsS)
# ax.scatter(q_x,y_test[:nx], marker= 'o', s=70, facecolors='none', edgecolors='tab:blue', label = 'actual')
# ax.scatter(q_x,y_pred[:nx], marker= 'x', s=70, color='tab:orange', label = 'predicted')
# ax.grid()
# plt.ylabel('M')
# plt.xlabel('quake')
# plt.legend(loc='upper left')
# plt.show()
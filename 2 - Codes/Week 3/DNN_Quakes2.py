# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 20:31:14 2020

@author: Ahmed_A_Torky
Dense Neural Network to Estimate PGA and IJMA
"""
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
# CPU Trainging is "-1". 
# Change to "0" or "1" only if you have a Nvidia GPU!
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.config.list_physical_devices('GPU')
print(tf.__version__)
# load the dataset
dataset = pd.read_csv('quakes_values.csv')
dataset_values = dataset.values
# Count Information in Data
Quake1 = dataset.loc[dataset['NumberOfSites'] == 1226]
# dataset['N-E'] = dataset[['Latitude(N)', 'Longitude(E)']].astype(str).agg('-'.join, axis=1)
# SensorsEN = pd.unique(dataset[['N-E']].values.ravel('K'))
# split into input (X) and output (y) variables
X = Quake1.values[:,[5,6,9]]
y = Quake1.values[:,[7,8]] # 7,8
# X = dataset_values[:,4:9]
# y = dataset_values[:,3] # [:,0:4]
try:
    if y.shape[1]:  
        ny = y.shape[1]
except IndexError:
    ny = 1
    y = np.reshape(y, [y.shape[0],1])
try:
    if X.shape[1]:  
        nX = X.shape[1]
except IndexError:
    nX = 1
    X = np.reshape(X, [X.shape[0],1])
# Scale X
scaler_X  = MaxAbsScaler()
scaler_X.fit(X)
X_scaled = scaler_X.transform(X)
# Scale Y
scaler_y  = MaxAbsScaler()
scaler_y.fit(y)
y_scaled = scaler_y.transform(y)
# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.20)
# Store training curves
loss = []
val_loss = []
ShortNameTr = []
ShortNameTe = []
# List of neurons in hidden layers
nodes_list = [20,30,50,100,200,500]
outnod_list = [10]
NumberOfRuns = len(nodes_list) * len(outnod_list)
# Loop on all neural networks
for i in range(len(nodes_list)):
    for j in range(len(outnod_list)):
        # Split
        nodes = nodes_list[i]
        outnod = outnod_list[j]
        print(nodes,outnod)
        # Clear previous memory
        tf.keras.backend.clear_session()
        tf.keras.backend.clear_session()
        tf.random.set_seed(123)
        
        # Define the model
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(nodes, input_dim=nX, activation='relu'))
        model.add(tf.keras.layers.Dense(nodes, activation='relu'))
        model.add(tf.keras.layers.Dense(outnod, activation='relu'))
        model.add(tf.keras.layers.Dense(ny, activation='relu'))
        # model summary
        model.summary()
        # optimizer choice
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,decay=0.0001) # learning_rate=0.001,decay=0.0001
        # compile the keras model
        model.compile(loss='mean_squared_error',
                          optimizer=optimizer, metrics=["accuracy"])
        # configure early stopping
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                              verbose=0, patience=300)
        # fit the keras model on the dataset
        history = model.fit(X_train, y_train, epochs=10000, batch_size=20,
                            shuffle=True, validation_split=0.33, verbose=1, 
                            callbacks=[es])
        
        # Make new preductions
        y_predict_test_scaled = model.predict(X_test)
        # Inverse scale X
        X_inverse_test = scaler_X.inverse_transform(X_test)
        # Inverse scale Y
        y_predict_test = scaler_y.inverse_transform(y_predict_test_scaled)
        y_inverse_test = scaler_y.inverse_transform(y_test)
        # evaluate the keras model
        print("Full dataset loss")
        _, accuracy = model.evaluate(X_scaled, y_scaled)
        print('Accuracy: %.2f' % (accuracy*100))
        print("Training dataset loss")
        _, accuracy = model.evaluate(X_train, y_train)
        print('Accuracy: %.2f' % (accuracy*100))
        print("Testing dataset loss")
        test_loss, accuracy = model.evaluate(X_test, y_test)
        print('Accuracy: %.2f' % (accuracy*100))
        print(test_loss)
        
        # Model Name
        Name = "Model: {"+np.str(nX)+", "+np.str(nodes)+", "+np.str(nodes)+ \
                    ", "+np.str(outnod)+", "+np.str(ny)+"} and Test MSE: "+ \
                    np.str('{:.5E}'.format(test_loss))
        ShortNameTr.append("Train: "+np.str(nodes)+", "+np.str(outnod))
        ShortNameTe.append("Test: "+np.str(nodes)+", "+np.str(outnod))
        # Append training curves to ciew them later
        loss.append(history.history['loss'])
        val_loss.append(history.history['val_loss'])
        # summarize history for loss
        fig = plt.figure(figsize=(10,7))
        fsS=12
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('Loss Curve - '+Name,fontsize=fsS)
        ax.plot(history.history['loss'], label = 'train')
        ax.plot(history.history['val_loss'], label = 'test')
        ax.set_yscale('log')
        ax.grid()
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend( loc='upper left')
        plt.show()
        
        # compare y_test (scatter plot)
        nx = 20
        q_x = np.linspace(1, nx, nx)
        # PGA
        fig = plt.figure(figsize=(10,7))
        fsS=12
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(Name,fontsize=fsS)
        ax.scatter(q_x,y_inverse_test[:nx,0], marker= 'o', s=70, facecolors='none', edgecolors='tab:blue', label = 'actual')
        ax.scatter(q_x,y_predict_test[:nx,0], marker= 'x', s=70, color='tab:orange', label = 'predicted')
        ax.grid()
        plt.ylabel('PGA')
        plt.xlabel('quake')
        plt.legend(loc='upper left')
        plt.show()
        # ijma
        fig = plt.figure(figsize=(10,7))
        fsS=12
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(Name,fontsize=fsS)
        ax.scatter(q_x,y_inverse_test[:nx,1], marker= 'o', s=70, facecolors='none', edgecolors='tab:blue', label = 'actual')
        ax.scatter(q_x,y_predict_test[:nx,1], marker= 'x', s=70, color='tab:orange', label = 'predicted')
        ax.grid()
        ax.set_ylim(0,7)
        plt.ylabel('IJMA')
        plt.xlabel('quake')
        plt.legend(loc='upper left')
        plt.show()
    
# summarize history for loss
fig = plt.figure(figsize=(10,7))
fsS=12
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Loss Curve',fontsize=fsS)
for i in range(NumberOfRuns):
    ax.plot(loss[i], label = ShortNameTr[i])
    ax.plot(val_loss[i], label = ShortNameTe[i])
ax.set_yscale('log')
ax.grid()
# ax.set_ylim(30**-3,10**-1)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()



# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 16:25:05 2017

@author: xws
"""
from keras.models import Sequential
from keras.layers import Dense, Dropout

## Generate dummy data
#x_train = np.random.random((1000, 20))
#y_train = np.random.randint(2, size=(1000, 1))
#x_test = np.random.random((100, 20))
#y_test = np.random.randint(2, size=(100, 1))

import pandas as pd
Train = pd.read_csv('C:/Users/xws/Desktop/challenger_stock/train.csv')
Test = pd.read_csv('C:/Users/xws/Desktop/challenger_stock/test.csv')


Y_train = Train['label'].values
X_train = Train.drop(['id','label','era','weight'],axis=1).values


X_test = Test.drop(['id'],axis=1).values


model = Sequential()
model.add(Dense(200, input_dim=89, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.fit(X_train, Y_train,
          epochs=200,
          batch_size=256)
score = model.predict(X_test)
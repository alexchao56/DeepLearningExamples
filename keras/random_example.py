#imports

import numpy as np
import random

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM, GRU

# set up model
features = 32
model = Sequential()
model.add(LSTM(features, 5))
model.add(Dropout(.2))
model.add(Dense(5, 1))
model.add(Activation('linear'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop')

# fake data
trials = 500
time_points = 100

X_train = np.random.rand(trials, time_points, features)
Y_train = np.arange(trials)#np.random.rand(100)

X_test = np.random.rand(trials, time_points, features)
Y_test = np.arange(trials)# np.random.rand(100)

for ind, y in enumerate(Y_train):
    X_train[y,5+ind/6,:6] = 100
    X_test[y,5+ind/6,:6] = 100

if False:
    for chan in range(3):
        for epoch in range(1,trials):
            X_train[epoch/2+5, epoch/trials, chan] = 10
            X_test[epoch/2+5, epoch/trials, chan] = 10

_ = plt.imshow(np.mean(X_test, 2))

model.fit(X_train, Y_train, batch_size=50, nb_epoch=10)
model.predict(X_test)
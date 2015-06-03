from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
from time import time

start = time()

train_path = "/home/ubuntu/data/cleanNeuralNetsData/train.csv"
train_target_path = "/home/ubuntu/data/cleanNeuralNetsData/train_target.csv"
test_path = "/home/ubuntu/data/cleanNeuralNetsData/test.csv"
test_target_path = "/home/ubuntu/data/cleanNeuralNetsData/test_target.csv"

# load data
print "Loading data"
print ""
X_train = np.loadtxt( train_path, delimiter = ',', skiprows=1 )
y_train = np.loadtxt( train_target_path, delimiter= ",", skiprows=1)
X_test = np.loadtxt( test_path, delimiter = ',', skiprows=1)
y_test= np.loadtxt( test_target_path, delimiter= ",", skiprows=1 )

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

input_dimension = X_train.shape[1]

print "Building a model"
print ""
model = Sequential()
model.add(Dense(input_dimension, input_dimension*2, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(input_dimension*2, input_dimension*2, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(input_dimension*2, 1, init='uniform'))

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

print "Fitting the model"
print ""
model.fit(X_train, y_train, nb_epoch=10, batch_size=(0.1 * X_train.shape[0]), verbose=2)

# calculate a mse score
score = model.evaluate(X_test, y_test, batch_size=16)

#Make predictions
print "Making predictions"
print ""
prediction = model.predict(X_test)

def calculate_rsquared(actual, predicted):
    from math import sqrt
    SSTot = sum(((actual-np.mean(actual)) ** 2))
    SSErr = sum((predicted - actual) ** 2)
    oosR2 = float(1 - (SSErr / SSTot))
    rmse = sqrt(SSErr / float(len(actual)))
    return "R-squared is: {0}, RMSE is: {1}".format(oosR2, rmse)

print calculate_rsquared(y_test, prediction)

end = time()

print "Total time elapsed: {0} seconds".format(end-start)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.layers.recurrent import SimpleDeepRNN
from keras.preprocessing import sequence
import numpy as np
from time import time
from theano import *

def create_columnwise_normalizers(nparray):
    """
    Takes in an input numpy array and normalizes/denormalizes it.
    Use if you need to normalize your data by translating to standard z scores.
    For the housing dataset, it makes more sense to normalize by dividing by the max.
    """

    try:
        feature_dimension = len(nparray[0,:])
        nrows = len(nparray[:,0])
    except IndexError:
        feature_dimension = 1
        nrows = len(nparray)

    column_mean = []
    column_std = []
    for i in range(feature_dimension):
        column_mean.append(np.mean(nparray[:,i]))
        column_std.append(np.std(nparray[:,i]))

    def normalize(nparray):
        temp = nparray
        start = time()
        for i in range(feature_dimension):
            temp[:,i] = (nparray[:,i] - column_mean[i]) / column_std[i]
        print "Scaling took {0} seconds".format(time() - start)
        return temp

    def denormalize(nparray):
        temp = nparray
        start = time()
        for i in range(feature_dimension):
            temp[:,i] = (nparray[:,i] * column_std[i]) + column_mean[i]
        print "Denormalizing took {0} seconds".format(time() - start)
        return temp

    return normalize, denormalize

start = time()

train_path = "../data/regression/housing_features_train.csv"
train_target_path = "../data/regression/housing_target_train.csv"
test_path = "../data/regression/housing_features_test.csv"
test_target_path = "../data/regression/housing_target_test.csv"
output_predictions_file = 'predictions.txt'

# load data
print "Loading data"
print ""
X_train = np.loadtxt( train_path, delimiter = ',', skiprows=1 )
y_train = np.loadtxt( train_target_path, delimiter= ",", skiprows=1)
X_test = np.loadtxt( test_path, delimiter = ',', skiprows=1)
y_test= np.loadtxt( test_target_path, delimiter= ",", skiprows=1 )

#Reshaping the y's
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

input_dimension = X_train.shape[1]

new_train = theano.shared(np.array((X_train, X_train, X_train)))
new_test = theano.shared(np.array((X_test, X_test, X_test)))
new_train = theano.shared(np.array((X_train, X_train, X_train)).reshape(399, 3, 13))
new_test = theano.shared(np.array((X_test, X_test, X_test)).reshape(107, 3, 13))


print "Building a model"
print ""
model = Sequential()
model.add(SimpleDeepRNN(input_dimension, input_dimension*2))
model.add(SimpleDeepRNN(input_dimension*2, input_dimension*2))
model.add(SimpleDeepRNN(input_dimension*2, input_dimension*2))
model.add(SimpleDeepRNN(input_dimension*2, input_dimension*2))
model.add(SimpleDeepRNN(input_dimension*2, input_dimension*2))
model.add(SimpleDeepRNN(input_dimension*2, 1, init = 'uniform', activation='tanh', inner_activation='tanh'))

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

print "Fitting the model"
print ""
model.fit(new_train, y_train / np.max(y_train), nb_epoch=100, batch_size=(0.1 * X_train.shape[0]), verbose=2)

# calculate a mse score
score = model.evaluate(new_test, y_test / np.max(y_test) /10, batch_size=0.1 * X_test.shape[1])

#Make predictions
print "Making predictions"
print ""
prediction = model.predict(X_test) * np.max(y_test)


print "Saving predictions"
np.savetxt( output_predictions_file, prediction, fmt = '%.6f' )

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

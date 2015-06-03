import numpy as np
import cPickle as pickle
from math import sqrt
from pybrain.datasets.supervised import SupervisedDataSet as SDS
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from time import time
from sklearn.metrics import mean_squared_error as MSE

def create_columnwise_normalizers(nparray):
    """ Takes in an input numpy array and normalizes/denormalizes it."""
    
    feature_dimension = len(nparray[0,:])
    nrows = len(nparray[:,0])
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
    
    def denomralize(nparray):
        temp = nparray
        start = time()
        for i in range(feature_dimension):
            temp[:,i] = (nparray[:,i] * column_std[i]) + column_mean[i]
        print "Denormalizing took {0} seconds".format(time() - start)
        return temp

    return normalize, denomralize

train_path = "../data/regression/housing_features_train.csv"
train_target_path = "../data/regression/housing_target_train.csv"
test_path = "../data/regression/housing_features_test.csv"
test_target_path = "../data/regression/housing_target_test.csv"
output_model_file_path = 'model.pkl'
output_predictions_file = 'predictions.txt'

hidden_size = 100
epochs = 30

print "Loading in the data"
train = np.loadtxt( train_path, delimiter = ',', skiprows=1 )
train_target = np.loadtxt( train_target_path, delimiter= ",", skiprows=1)
test = np.loadtxt( test_path, delimiter = ',', skiprows=1)
test_target = np.loadtxt( test_target_path, delimiter= ",", skiprows=1 )

train_target = train_target.reshape(-1, 1)
input_size = train.shape[1]
target_size = train_target.shape[1]

# prepare dataset
print "Preparing the dataset"
print ""
ds = SDS( input_size, target_size )
ds.setField( 'input', train )
ds.setField( 'target', train_target / np.max(train_target) )

# init and train
print "Initalizing the network and training"
net = buildNetwork( input_size, hidden_size, target_size, bias = True )
trainer = BackpropTrainer( net,ds )

start = time()
for i in range( epochs ):
    mse = trainer.train()
    rmse = sqrt( mse )
    print "training RMSE, epoch {}: {}".format( i + 1, rmse )
end = time()
print "Training took: " + str((end - start)) + "seconds"
print ""

print "Pickling the model"
pickle.dump( net, open( output_model_file_path, 'wb' ))

print "Preparing to test the model"
test_target = test_target.reshape(-1, 1)
input_size = test.shape[1]
target_size = test_target.shape[1]

#load model
print "Loading the pickled model"
net = pickle.load( open( output_model_file_path, 'rb' ))

assert( net.indim == input_size )
assert( net.outdim == target_size )

# prepare dataset
ds = SDS( input_size, target_size )
ds.setField( 'input', test )
ds.setField( 'target', test_target / np.max(test_target) )


# predict
print "Predicting..."
predicted = net.activateOnDataset( ds ) * np.max(test_target)
	
mse = MSE( test_target, predicted )
rmse = sqrt( mse )

print "testing RMSE:", rmse

print "Saving predictions"
np.savetxt( output_predictions_file, predicted, fmt = '%.6f' )


def calculate_rsquared(actual, predicted):
    SSTot = sum(((actual-np.mean(actual)) ** 2))
    SSErr = sum((predicted - actual) ** 2)
    oosR2 = float(1 - (SSErr / SSTot))
    rmse = sqrt(SSErr / float(len(actual)))
    return "R-squared is: {0}, RMSE is: {1}".format(oosR2, rmse)

print calculate_rsquared(test_target, predicted)
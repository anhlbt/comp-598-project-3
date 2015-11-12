from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
import data_manager
import numpy as np
from nolearn.lasagne import NeuralNet
import lasagne
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import adam
from lasagne.layers import get_all_params
from lasagne.updates import nesterov_momentum

from sklearn.metrics import classification_report
from sklearn.grid_search import RandomizedSearchCV


from scipy.stats import randint as sp_randint
from scipy.stats import expon
from scipy.stats import uniform

import theano
import pickle
from time import time
from operator import itemgetter

def formatData(XDATA, yDATA = None):
    # apply some very simple normalization to the data
    XDATA -= XDATA.mean()
    XDATA /= XDATA.std()

    XDATA = XDATA.reshape(-1,1,48,48)
    
    #X_train,  X_val, y_train, y_val = train_test_split(XDATA,yDATA,test_size=10000,random_state=42)

    #X_test, X_val,y_test,y_val = train_test_split(temp_set_x,temp_set_y,test_size=.5,random_state=42)

    X_train = XDATA.astype(np.float32)
    #X_test = X_test.astype(np.float32)
    #X_val = X_val.astype(np.float32)

    y_train = yDATA.astype(np.int32)
    #y_test = y_test.astype(np.int32)
    #y_val = y_val.astype(np.int32)

    return X_train,y_train

def float32(k):
    return np.cast['float32'](k)

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

# Utility function to report best scores
def report(grid_scores, n_top=10):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    S=''
    for i, score in enumerate(top_scores):
        S+=("Model with rank: {0}".format(i + 1))
        S+=("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        S+=("Parameters: {0}".format(score.parameters))
        S+=("")
    return S

		
convnet = NeuralNet(
	

	layers = [
	    (InputLayer, {'shape': (None, 1, 48,48)}),

	    (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
	   
	    (MaxPool2DLayer, {'pool_size': (2, 2)}),
	    (DropoutLayer, {'p':.2}),

	    (DenseLayer, {'num_units': 256}),
	    (DropoutLayer, {'p':.5}),
	    (DenseLayer, {'num_units': 256}),

	    (DenseLayer, {'num_units': 10, 'nonlinearity': softmax}),
	],
	update_learning_rate=theano.shared(float32(.01)),
	update_momentum=theano.shared(float32(0.9)),
	#update_learning_rate=.01,
	verbose=0,
	max_epochs = 100,


)



#load data
print("Loading data...")
XDATA, yDATA = data_manager.load_test_data()

#format data
X_train,y_train = formatData(XDATA,yDATA)

#Take 10% of the data
X_train, _, y_train, _= train_test_split(X_train, y_train, test_size=0.90, random_state=42)

#split into 80/20 train test
cv = StratifiedShuffleSplit(y_train, n_iter=1,test_size=0.2, random_state=42)


parameters = {'num_filters':sp_randint(8,64),
			'update_learning_rate':uniform(loc=0.001,scale = 0.01),
			'update_momentum': uniform(loc = 0.5,scale= 0.45),
			'p':uniform(loc = 0.1,scale = 0.6),
			}


grid_search = RandomizedSearchCV(convnet,parameters,cv=cv,verbose = 2,n_iter=100)


grid_search.fit(X_train, y_train)

start = time()

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.grid_scores_)))
report(grid_search.grid_scores_)

report1 = report(grid_search.grid_scores_)

with open("gridrep.txt", "w") as text_file:
    text_file.write(report1)

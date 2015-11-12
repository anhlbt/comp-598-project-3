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

import theano


from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import objective

import numpy as np
import pickle

import data_manager

#visualizations
#from nolearn.lasagne.visualize import plot_loss
#from nolearn.lasagne.visualize import plot_conv_weights
#from nolearn.lasagne.visualize import plot_conv_activity
#from nolearn.lasagne.visualize import plot_occlusion
#from nolearn.lasagne import PrintLayerInfo
import sys

sys.setrecursionlimit(10000)


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


convnet = NeuralNet(
    layers = [
        (InputLayer, {'shape': (None, 1, 38,38)}),

        (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
        #(Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
        #(Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
        
        (MaxPool2DLayer, {'pool_size': (2, 2)}),
        (DropoutLayer, {'p':.5}),

        (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 1}),
        #(Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 1}),
       	#(Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 1}),

        (MaxPool2DLayer, {'pool_size': (2, 2)}),
        (DropoutLayer, {'p':.5}),

        (DenseLayer, {'num_units': 256}),
        (DropoutLayer, {'p':.5}),
        (DenseLayer, {'num_units': 256}),

        (DenseLayer, {'num_units': 10, 'nonlinearity': softmax}),
    ],
    update_learning_rate=theano.shared(float32(0.005)),
    update_momentum=theano.shared(float32(0.9)),
    #update_learning_rate=.01,
    verbose=2,
    max_epochs = 200,
    
    )

def formatData(XDATA, yDATA = None):
    # apply some very simple normalization to the data
    XDATA -= XDATA.mean()
    XDATA /= XDATA.std()

    XDATA = XDATA.reshape(-1,1,38,38)

    X_train = XDATA.astype(np.float32)
    

    y_train = yDATA.astype(np.int32)
    

    return X_train,y_train

# Load the dataset
print("Loading data...")

XDATA = np.load('rotatedTrimedX.npz')['arr_0']
yDATA = np.load('rotatedTrimedY.npz')['arr_0']

#XDATA, yDATA = data_manager.load_all_images(True,True)
#XDATA, yDATA = data_manager.load_transformed_images()
X_train,y_train = formatData(XDATA,yDATA)

print(X_train.shape)
convnet.fit(X_train,y_train)

with open('./CNNMODELS/convnet.pickle', 'wb') as f:
    pickle.dump(convnet, f, -1)






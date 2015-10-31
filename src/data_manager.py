__author__ = "Charlie"


import numpy as np

train_inputs1 = './data/train_inputs1.npz'
train_inputs2 = './data/train_inputs2.npz'
train_outputs = './data/train_outputs.npz'
test_inputs   = './data/test_inputs.npz'

def load_test_data():
        X1 = np.load(train_inputs1)['train_inputs1_np']
        X2 = np.load(train_inputs2)['train_inputs2_np']
        X  = np.vstack((X1,X2))
        Y  = np.load(train_outputs)['train_outputs_np']
        return X,Y

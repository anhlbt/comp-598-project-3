import unittest, random

from neural_net import NeuralNet as NN
from utilities import *

import numpy as np

random.seed(123)
np.random.seed(123)

class TestNN(unittest.TestCase):

    def test_neuronize(self):
        y=[[2],[0],[1]]
        result=neuronize(y)
        expected=[[0,0,1],[1,0,0],[0,1,0]]
        self.assertEqual(result,expected)

    def test_weight_dimensions(self):
        nn=NN((2,2,1),verbose=0)

        self.assertEqual(nn.weights[1].shape,(2,3))
        self.assertEqual(nn.weights[2].shape,(1,3))

    def test_output_dimensions(self):
        nn=NN((2,2,1),verbose=0)
        inputs=([1,0],[0,1],[1,1],[0,0])
        for i in inputs:
            nn.forward(i)
            self.assertEqual(nn.outputs[0].shape,(3,1))
            self.assertEqual(nn.outputs[1].shape,(3,1))
            self.assertEqual(nn.outputs[2].shape,(1,1))
            self.assertEqual(nn.get_output().shape,(1,))
            
    def test_backward(self):
        X=[[0,0],[0,1],[1,0],[1,1]]
        Y=[[0],[1],[1],[0]]
        nn=NN([2,2,1],verbose=0)
        for i in range(4):
            nn.forward(X[i])
            nn.backward(Y[i])
            self.assertEqual(nn.outputs[0].shape,(3,1))
            self.assertEqual(nn.outputs[1].shape,(3,1))
            self.assertEqual(nn.outputs[2].shape,(1,1))
            self.assertEqual(nn.get_output().shape,(1,))

    def test_1_hidden_2n_xor(self):
        X=[[0,0],[0,1],[1,0],[1,1]]
        Y=[[0],[1],[1],[0]]
        nn=NN([2,2,1],verbose=0,learning_rate=0.2)
        nn.train(X,Y,10000)
        report=nn.get_report(X,Y)
        self.assertEqual(report["errors"],[])

    def test_1_hidden_6n_xor(self):
        X=[[0,0],[0,1],[1,0],[1,1]]
        Y=[[0],[1],[1],[0]]
        nn=NN([2,6,1],verbose=0,learning_rate=0.2)
        nn.train(X,Y,10000)
        report=nn.get_report(X,Y)
        self.assertEqual(report["errors"],[])

    def test_3_bools(self):
        X,Y,a,b=get_data("tests/3bools.csv",1)

        nn=NN([3,10,1],verbose=0,learning_rate=0.1)
        nn.train(X,Y,10000)
        report=nn.get_report(X,Y)
        self.assertEqual(report["errors"],[])

    def test_6_bools(self):
        X,Y,a,b=get_data("tests/6bools.csv",1)

        nn=NN([6,36,1],verbose=0,learning_rate=0.1)
        nn.train(X,Y,10000)
        report=nn.get_report(X,Y)
        self.assertEqual(report["errors"],[])

if __name__=="__main__":
    unittest.main()

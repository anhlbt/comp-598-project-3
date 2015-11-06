import unittest, random

from neural_net import NeuralNet as NN
from utilities import *

random.seed(123)

class TestNN(unittest.TestCase):

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

    def test_1_hidden_xor(self):
        X=[[0,0],[0,1],[1,0],[1,1]]
        Y=[[0],[1],[1],[0]]
        nn=NN([2,2,1],verbose=0,learning_rate=0.2)
        for i in range(10000):
            index=random.randint(0,3)
            nn.forward(X[index])
            nn.backward(Y[index])
        
        for i in range(4):
            nn.forward(X[i])
            nn.backward(Y[i])
            
            result=1 if nn.get_output()[0]>0.5 else 0
            #print(nn.get_output())
            self.assertEqual(result,Y[i][0],msg="%s: "%i+str(X[i]))

    def test_2_hidden_xor(self):
        X=[[0,0],[0,1],[1,0],[1,1]]
        Y=[[0],[1],[1],[0]]
        nn=NN([2,2,2,1],verbose=0,logging=True,learning_rate=0.2)
        for i in range(10000):
            index=random.randint(0,3)
            nn.forward(X[index])
            nn.backward(Y[index])
        
        for i in range(4):
            nn.forward(X[i])
            nn.backward(Y[i])
            
            result=1 if nn.get_output()[0]>0.5 else 0
            #print(nn.get_output())
            self.assertEqual(result,Y[i][0],msg="%s: "%i+str(X[i]))

if __name__=="__main__":
    unittest.main()

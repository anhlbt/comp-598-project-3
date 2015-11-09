 
import numpy as np
import os, random
from utilities import *
from constants import *
from neural_net_view import NeuralNetView

class NeuralNet(NeuralNetView):  
    def __init__(self, sizes,learning_rate=0.1,verbose=0,logging=0,timer_interval=10):
        self.sizes=sizes
        self.learning_rate = learning_rate
        self.verbose=verbose
        self.back_count=0
        self.logging=logging
        self.timer_interval=timer_interval
        if logging:
            self.setup_logging()
        
        #initialize all the lists.
        #every item in the following four lists are numpy arrays
        self.weights=[]
        self.outputs=[]
        #activations and corrections are assigned before they are used
        #so we can just initialize a list of the correct size
        self.activations=[-1 for i in sizes]
        self.corrections=[-1 for i in sizes]

        previous_ncount=0
        for i,ncount in enumerate(sizes):
            #initialize all weights randomly
            self.weights.append(np.random.random((ncount,previous_ncount)))
            #add one to ncount for bias neurons
            previous_ncount=sizes[i]+1

            #all activations, outputs, and corrections start at zero
            is_last=i==len(sizes)-1
            self.outputs.append(np.zeros((ncount+(0 if is_last else 1),1),dtype=float))

    def activation_func(self,x):
        return np.tanh(x)

    def d_activation_func(self,x):
        return self.activation_func(x)*(1-self.activation_func(x))
                            
    def forward(self, inputs):
        #where inputs is simply a list of numbers, of length self.sizes[0]
        if self.verbose>1:
            print_color("Starting forward.",COLORS.GREEN)

        self.outputs[0][:-1, 0] = inputs
        self.outputs[0][-1:, 0] = 1.0

        for i in range(1,len(self.sizes)):
            is_last=i==len(self.sizes)-1

            self.activations[i] = np.dot(self.weights[i], self.outputs[i-1])
            if is_last:
                #the last layer does not have a bias neuron to ignore, thus the if statement
                self.outputs[i] = self.activation_func(self.activations[i])
            else:
                self.outputs[i][:-1, :] = self.activation_func(self.activations[i])

            #set bias neuron to always output 1
            if not is_last:
                self.outputs[i][-1:, :] = 1.0
    
    def backward(self, desired_outputs):
        #where desired_outputs is simply a list of numbers 0 to 1, of length self.sizes[-1]
        self.back_count+=1
        if self.verbose>1:
            print_color("Starting backward.",COLORS.ORANGE)

        desired_array=np.array(desired_outputs,dtype=float)
        desired_array=desired_array.reshape(desired_array.shape[0],1)
        error = self.outputs[-1] - desired_array
        assert error.shape == self.outputs[-1].shape
        for i in reversed(range(1,len(self.sizes))):
            is_last=i==len(self.sizes)-1

            if is_last:
                #the last calculation uses an error based on desired_outputs, instead of
                #the next layer, since there is no next layer
                self.corrections[i] = self.d_activation_func(self.activations[i]) * error
            else:
                #if not last layer, get propagate error from next layer
                self.corrections[i] = self.d_activation_func(self.activations[i]) * np.dot(self.weights[i+1][:,:-1].transpose(), self.corrections[i+1])

            #adjust weights according to those corrections
            self.weights[i] = self.weights[i] - self.learning_rate * np.dot(self.corrections[i],
                    self.outputs[i-1].transpose()) 
        
        if self.logging:
            self.log()
    
    def get_output(self):
        #where output is the output of the final layer
        return self.outputs[-1].flatten()

    def train(self,X,Y,trial_count):
        if len(X) != len(Y) or len(X[0]) != self.sizes[0]:
            raise ValueError("NeuralNetwork.train got weird X or Y data. len(X)=%s len(Y)=%s len(X[0])=%s sizes[0]=%s"%(
                len(X),len(Y),len(X[0]),self.sizes[0]))
        if self.verbose:
            print_color("Started training for %s trials."%trial_count,COLORS.YELLOW)

        #convert Y=[[2],[0],[1]...] to Y=[[0,0,1],[1,0,0],[0,1,0]...], or does nothing if Y=[[1],[0] ...]
        Y=neuronize(Y)

        timer=Timer(self.timer_interval)
        for i in range(trial_count):
            if self.verbose:
                timer.tick("Running trial %s/%s."%(i,trial_count))
            index=random.randint(0,len(X)-1)
            self.forward(X[index])
            self.backward(Y[index])
        if self.verbose:
            timer.stop("Training")

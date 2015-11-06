 
import numpy as np
import os
from utilities import *
from constants import *

np.random.seed(123)

class NeuralNet:  
    def __init__(self, sizes,learning_rate=0.1,verbose=0,logging=0):
        self.sizes=sizes
        self.learning_rate = learning_rate
        self.verbose=verbose
        self.logging=logging
        if logging:
            self.log_filename="weights.log"
            try:
                os.remove(self.log_filename)
            except FileNotFoundError:
                pass
        self.back_count=0
        
        #initialize all the lists.
        #every item in the following four lists are numpy arrays
        self.weights=[]
        self.outputs=[]
        #activations and corrections are assigned before they are used
        #so we can just initialize a list of the correct size
        self.activations=[0 for i in sizes]
        self.corrections=[0 for i in sizes]

        previous_ncount=0
        for i,ncount in enumerate(sizes):
            #initialize all weights randomly
            self.weights.append(np.random.random((ncount,previous_ncount)))
            #add one to ncount for bias neurons
            previous_ncount=sizes[i]+1

            #all activations, outputs, and corrections start at zero
            is_last=i==len(sizes)-1
            self.outputs.append(np.zeros((ncount+(0 if is_last else 1),1),dtype=float))
            
    def show(self,weights=False,outputs=False,activations=False,corrections=False,all=False):
        #this is a convenient way to show some or all of the NN info
        def show_np_list(label,np_list):
            print_color("%s:"%label,COLORS.YELLOW)
            for i,item in enumerate(np_list):
                s="" if type(item) is int else str(item.shape)
                print("%s:"%i,s,item)
        
        if weights or all:
            show_np_list("weights",self.weights)
        if outputs or all:
            show_np_list("outputs",self.outputs)
        if activations or all:
            show_np_list("activations",self.activations)
        if corrections or all:
            show_np_list("corrections",self.corrections)

    def activation_func(self,x):
        return np.tanh(x)

    def d_activation_func(self,x):
        return self.activation_func(x)*(1-self.activation_func(x))
                            
    def forward(self, inputs):
        #where inputs is simply a list of numbers, of length self.sizes[0]
        if self.verbose:
            print_color("Starting forward.",COLORS.GREEN)
            self.show(all=True)

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
        #where desired_outputs is simply a list of numbers, of length self.sizes[-1]
        self.back_count+=1
        if self.verbose:
            print_color("Starting backward.",COLORS.ORANGE)
            self.show(all=True)

        error = self.outputs[-1] - np.array(desired_outputs, dtype=float)
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

    def log(self):
        text=[]
        for w in self.weights:
            text+=w.flatten().tolist()
        text=",".join([str(round(i,3)) for i in text])
        with open(self.log_filename,"a") as f:
            f.write("\n"+text)
    
    def get_output(self):
        #where output is the output of the final layer
        #[0] is so that it returns a vector, not a weirdo 2D array with one column
        return self.outputs[-1][0]




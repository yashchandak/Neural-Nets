# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 17:02:55 2015

Multilayer backpropagation neural network

[       ]       [       ]       [       ]       [       ]
[Input  ]       [Sigmoid]       [sigmoid]       [       ]
[vector ] ----> [hidden ] ----> [hidden ] ----> [Output ]
[       ]       [layer 1]       [layer 2]       [       ]
[       ]       [       ]       [       ]       [       ]
 
 ----> full connections


TODO:
1) convert to modular class/object based design
2) addition of biases
3) generalise to n number of hidden layers
4) optimisation
        a) momentum
        b) conjugated gradient descent
        
5) normalise input and output data
6) simulated annealing - decaying learning parameter/step size
7) cache constant intermediate results instead of recalculating
8) end training based on difference from prv error
9) 

@author: yash and ankit
"""

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np


def error_plot():
    plt.plot(range(iter_no), err)
    plt.xlabel('iteration number')
    plt.ylabel('error value')
    plt.title('Error PLot')
    plt.show()    
    

def generate_data():
    return 0

    
def train_nets():
    global layer1, layer2, layer3, err
    
    error = 0
    for index in xrange(iter_no):
        #update based on each data point
        for i in len(data):
            inputs, expected = 0 #parse the data to get input and expected output
            
            execute_net(inputs)
            error = output - expected   #error vector corresponding to each output
            
            #update weights between 2nd hidden network and outputs
            for k in range(nodes_hidden2):
                for l in range(nodes_output):
                    layer3[l][k] += learning_rate*error[l]*(output[l]*(1-output[l]))*hidden2[k]
                    
            #update weights between 1st and 2nd hidden layers
            for j in range(nodes_hidden1):  
                for k in range(nodes_hidden2):
                    for l in range(nodes_output):
                        layer2[k][j] += learning_rate*error[l]*(output[l]*(1-output[l]))*layer3[l][k]*(hidden2[k]*(1-hidden2[k]))*hidden1[j]
                        
            #update weights between input and 1st hidden layer
            for n in range(inp):
                for j in range(nodes_hidden1):  
                    for k in range(nodes_hidden2):
                        for l in range(nodes_output):
                            layer1[j][n] += learning_rate*error[l]*(output[l]*(1-output[l]))*layer3[l][k]*(hidden2[k]*(1-hidden2[k]))*layer2[k][j]*(hidden1[j]*(1-hidden1[j]))*inputs[n]
            
        err[index] = error

    
def execute_net(inputs):
    global hidden1, hidden2, output
    
    #all layers have sigmoidal activation function
    for i in range(nodes_hidden1):
        hidden1[i] = 1/(1 + e**(layer1[i][:].dot(inputs)))
        
    for i in range(nodes_hidden2):
        hidden2[i] = 1/(1 + e**(layer2[i][:].dot(hidden1)))
        
    for i in range(nodes_output):
        output[i] = 1/(1 + e**(layer3[i][:].dot(hidden2)))
        

    
"""
NETWORK TOPOLOGY
"""
e = 2.718281828
inp = 2                     #input vector dimensions:
nodes_hidden1 = 10          #number of nodes in hidden layer 1
nodes_hidden2 = 5           #number of nodes in hidden layer 2
nodes_output = 1            #number of outputs
learning_rate = 0.01
iter_no = 5000              #training iterations


"""
DATA generation, internediate values and weights initialisation
"""
data = generate_data()                                      #get data
err = np.zeros(iter_no)

hidden1 = np.zeros(nodes_hidden1, 'float')                  #hidden layer 1
hidden2 = np.zeros(nodes_hidden2, 'float')                  #hidden layer 2
output = np.zeros(nodes_output, 'float')                    #output layer

layer1 = np.random.random((nodes_hidden1,inp))              #weights b/w input and first hidden layer nodes
layer2 = np.random.random((nodes_hidden2,nodes_hidden1))    #weights b/w hidden layer nodes
layer3 = np.random.random((nodes_output,nodes_hidden2))     #weights b/w second hidden layer and outputs


    
def start():
    return 0
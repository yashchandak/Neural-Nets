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
9) Matrix notation for weight updates
10)intermediate storage of weights in some file (for recovery/comparison)
11)bias weights

@author: yash and ankit
"""

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np


def error_plot():
    plt.figure(1)
    plt.plot(range(iter_no), err)
    plt.xlabel('iteration number')
    plt.ylabel('error value')
    plt.title('Error PLot')
    plt.show()    
    

def generate_data():
    #data = [[.1,.1,0.75],[.4,.1,0.75],[.1,.4,0.75],[.1,.7,0],[.1,.9,0], [.3,.8,0], [.7,.7,0.25],[.6,.99,0.25],[.9,.7,0.25],[.8,.1,1],[.7,.3,1],[.99,.2,1],[.5,.5,0.5],[.8,.5,0.5], [.4,.7,0.5], [.6,.8,0.5]]
    #data = [[.1,.1,0,0],[.4,.1,0,0],[.1,.4,0,0],[.1,.7,0,1],[.1,.9,0,1], [.3,.8,0,1], [.7,.7,1,0],[.6,.99,1,0],[.9,.7,1,0],[.8,.1,1,1],[.7,.3,1,1],[.99,.2,1,1]]
    #data = [[0,0,0],[0,1,1],[1,0,1],[1,1,0]] #XOR   
    data = np.array([[0,0],[1,1],[4,2],[5,4],[6,6],[7,7],[8,9],[9,9],[12,10],[13,9],[15,8],[16,7],[20,4],[21,2],[23,0],[24,0]], dtype = 'float')
    
    plt.figure(0)
    plt.plot(data[:,0],data[:,1],'b+')      #data plot

    max_x = np.amax(data[:,0])
    max_y = np.amax(data[:,1])
    min_x = np.amin(data[:,0])
    min_y = np.amin(data[:,1])
    
    #normalise the data
    data[:,0] = (data[:,0]-min_x)/(max_x-min_x)
    data[:,1] = (data[:,1]-min_y)/(max_y-min_y)    
    
    return data

def get_data(index):
    #in1, in2, out1, out2 = data[index]             #parse the data to get input and expected output
    in1, out1 = data[index]    
    return np.array([in1]),np.array([out1])
    
def train_nets():
    global layer1, layer2, layer3, err
    
    error = 0
    for index in xrange(iter_no):
        #update based on each data point
        for i in xrange(len(data)):
            inputs, expected = get_data(i)
            execute_net(inputs)
            error = expected - output   #error vector corresponding to each output
            
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
            
        err[index] = sum(abs(error))
        if index%100 == 0:
            print "Iteration no: ", index, "    error: ", error

    
def execute_net(inputs):
    global hidden1, hidden2, output
    
    #all layers have sigmoidal activation function
    hidden1 = 1/(1 + e**-(layer1.dot(inputs)))
    hidden2 = 1/(1 + e**-(layer2.dot(hidden1)))
    output  = 1/(1 + e**-(layer3.dot(hidden2)))
    
    """    
    for i in range(nodes_hidden1):
        hidden1[i] = 1/(1 + e**-(layer1[i][:].dot(inputs)))
        
    for i in range(nodes_hidden2):
        hidden2[i] = 1/(1 + e**-(layer2[i][:].dot(hidden1)))
        
    for i in range(nodes_output):
        output[i] = 1/(1 + e**-(layer3[i][:].dot(hidden2)))
    """    

    
"""
NETWORK TOPOLOGY
"""
e = 2.718281828
inp = 1                     #input vector dimensions:
nodes_hidden1 = 3          #number of nodes in hidden layer 1
nodes_hidden2 = 2           #number of nodes in hidden layer 2
nodes_output  = 1           #number of outputs
learning_rate = 0.1
iter_no = 1000              #training iterations


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
    train_nets()
    error_plot()
    
    while(1):
        x = input()
        if x > 99:
            break
        execute_net([x])
        print "output: ", output
    return 0
    
start()
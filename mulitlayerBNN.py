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
7) [DONE] cache constant intermediate results instead of recalculating
8) end training based on difference from prv error
9) [DONE] Matrix notation for weight updates
10)intermediate storage of weights in some file (for recovery/comparison)

@author: yash and ankit
"""

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import time
#import dataset

def generate_data():

    data = [[0,0,0,0,0],[0,0,0,1,1],[0,0,1,0,1],[0,0,1,1,0],[0,1,0,0,1],[0,1,0,1,0],[0,1,1,0,0],[0,1,1,1,1],[1,0,0,0,1],[1,0,0,1,0],[1,0,1,0,0],[1,0,1,1,1],[1,1,0,0,0],[1,1,0,1,1],[1,1,1,0,1],[1,1,1,1,0]]
    #data = [[.1,.1,0.75],[.4,.1,0.75],[.1,.4,0.75],[.1,.7,0],[.1,.9,0], [.3,.8,0], [.7,.7,0.25],[.6,.99,0.25],[.9,.7,0.25],[.8,.1,1],[.7,.3,1],[.99,.2,1],[.5,.5,0.5],[.8,.5,0.5], [.4,.7,0.5], [.6,.8,0.5]]
    #data = [[.1,.1,0,0],[.4,.1,0,0],[.1,.4,0,0],[.1,.7,0,1],[.1,.9,0,1], [.3,.8,0,1], [.7,.7,1,0],[.6,.99,1,0],[.9,.7,1,0],[.8,.1,1,1],[.7,.3,1,1],[.99,.2,1,1]]
    #data = [[0,0,0],[0,1,1],[1,0,1],[1,1,0]] #XOR   
    #data = [[0,0,0],[0,1,1],[1,0,1],[1,1,0]] #AND
    #data = np.array([[0,0],[1,1],[4,2],[5,4],[6,6],[7,7],[8,9],[9,9],[12,10],[13,9],[15,8],[16,7],[20,4],[21,2],[23,0],[24,0]], dtype = 'float')
    #data = np.array([[.1,.2,.3,.1],[.3,.5,.8,.2]])    #manual test data
    
    return data

def get_data(index):
    #in1, in2, out1 = data[index]             #parse the data to get input and expected output
    #in1, out1 = data[index]    
    in1, in2, in3, in4, out1 = data[index]
    return np.array([in1,in2, in3, in4]),np.array([out1])


def activate(z, derivative = False):
    #Sigmoidal activation function
    if derivative:
        return z*(1-z)        
    return 1/(1+e**-z)
    
    #tanh activation function
    
def plotit(x,y, fig, xlabel, ylabel, title):
    plt.figure(1)
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()    
    
    
def train_nets():
    global layer1, layer2, layer3, err, del1, del2, del3, deltas, synapses
    
    error = 0    
    for epoch in xrange(iter_no):
        #update based on each data point
        error_sum = 0
        for i in xrange(len(data)):            
            
            inputs, expected = get_data(i)
            execute_net(inputs)
            error = expected - receptors[depth-1]   #error vector corresponding to each output
            error_sum += sum(abs(error))
                     
            deltas[depth-1] = activate(receptors[depth-1],True)*error
            for index in xrange(depth-2, -1, -1):
                deltas[index] = activate(receptors[index],True)*synapses[index+1].transpose().dot(deltas[index+1])
            
            #update all the weights
            for index in xrange(depth-1, 0, -1):
                synapses[index] += learning_rate*deltas[index].reshape(topology[index+1],1)*receptors[index-1]
            synapses[0] += learning_rate*deltas[0].reshape(topology[1],1)*inputs
           
        err[epoch] = error_sum/len(data)
        if epoch%1000 == 0:
            print "Iteration no: ", epoch, "    error: ", err[epoch]

    
def execute_net(inputs):
    global hidden1, hidden2, output, synapses, receptors

    #activate the nodes based on sum of incoming synapses    
    receptors[0] = activate(synapses[0].dot(inputs)) #activate first time based on inputs
    for index in xrange(1,depth):        
        receptors[index] = activate(synapses[index].dot(receptors[index-1]))
    
"""
NETWORK TOPOLOGY
"""
e = 2.718281828
inp = 4                 #input vector dimensions:
nodes_output  = 1  #number of outputs
learning_rate = 0.5
momentum = 0.3
iter_no = 25000              #training iterations


"""
DATA generation, internediate values and weights initialisation
"""
data = generate_data()                                      #get data
err = np.zeros(iter_no)

topology = np.array([inp,20,8,nodes_output])
depth = topology.size - 1

synapses = [np.random.random((size2,size1)) for size1,size2 in zip(topology[0:depth],topology[1:depth+1])]
receptors = [np.zeros(size, 'float') for size in topology[1:]] #does not have inputs
deltas = [np.zeros(size, 'float') for size in topology[1:]]    


def main():
    train_nets()
    plotit(range(iter_no), err, 1, 'iteration number', 'error value', 'Error PLot')
    
#    while(1):
#        v,w,x,y = input()
#        if v > 99:
#            break
#        execute_net([v,w,x,y])
#        print "output: ", output
#    return 0
 
start = time.clock()   
main()
end = time.clock()
print 'time elapsed: ', end-start
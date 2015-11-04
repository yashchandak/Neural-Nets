# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 17:02:55 2015

Multilayer backpropagation neural network

[       ]       [       ]       [       ]       [       ]       [       ]
[Input  ]       [Sigmoid]       [sigmoid]       [sigmoid]       [       ]
[vector ] ====> [hidden ] ----> [hidden ] ----> [hidden ] ====> [Output ]
[       ]       [layer 1]       [layer  ]       [layer N]       [       ]
[       ]       [       ]       [  ...  ]       [       ]
 
 ----> full connections

priyadarshini.j
TODO:
1) convert to modular class/object based design
2) *addition of biases
3) [DONE] generalise to n number of hidden layers
4) *optimisation
        a) momentum
        b) conjugated gradient descent  
        c) regularisation
5) normalise input and output data
6) *simulated annealing - decaying learning parameter/step size
7) [DONE] cache constant intermediate results instead of recalculating
8) end training based on difference from prv error
9) [DONE] Matrix notation for weight updates
10)*intermediate storage of weights in some file (for recovery/comparison)
11) geometric image manip
12)GPU usage
13)plotting error function and output for 2d/3d values

@author: yash and ankit
"""

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import time
import dataset

"""
NETWORK TOPOLOGY
"""
e = 2.718281828
inp = dataset.inp                 #input vector dimensions:
nodes_output  = dataset.output  #number of outputs
learning_rate = 0.5
momentum = 0.3
iter_no = 3000              #training iterations

"""
DATA generation, internediate values and weights initialisation
"""
data = dataset.data                                      #get data
test = dataset.test
err = np.zeros(iter_no)
test_err = np.zeros(iter_no)

topology = np.array([inp,16,nodes_output])
depth = topology.size - 1

synapses = [np.random.random((size2,size1)) for size1,size2 in zip(topology[0:depth],topology[1:depth+1])]
receptors = [np.zeros(size, 'float') for size in topology[1:]] #does not have inputs
deltas = [np.zeros(size, 'float') for size in topology[1:]]   


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
    global  err, test_err, deltas, synapses
    
    error = 0    
    for epoch in xrange(iter_no):
        #update based on each data point
        error_sum = 0
        test_error_sum = 0
        for i in xrange(len(data)):            
            
            inputs, expected = dataset.get_data(i)
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
         
         
        for i in xrange(len(test)):
            inputs, expected = dataset.get_test(i)
            execute_net(inputs)
            
            tt = np.zeros(nodes_output)
            pos = np.argmax(receptors[depth-1])
            tt[pos] = 1            
                
            test_error_sum += sum(abs(expected - tt))
            #test_error_sum += sum(abs(expected - receptors[depth-1]))
        
        err[epoch] = error_sum/len(data)
        test_err[epoch] = test_error_sum/(2*len(test)) #single mis classification creates an error sum of 2.
        
        if epoch%100 == 0:
            print "Iteration no: ", epoch, "    error: ", err[epoch], " test error: ", test_err[epoch]

    
def execute_net(inputs):
    global synapses, receptors

    #activate the nodes based on sum of incoming synapses    
    receptors[0] = activate(synapses[0].dot(inputs)) #activate first time based on inputs
    for index in xrange(1,depth):        
        receptors[index] = activate(synapses[index].dot(receptors[index-1]))
     


def main():
    while(1):
        train_nets()
        plotit(range(iter_no), err, 1, 'iteration number', 'error value', 'Error PLot')
        plotit(range(iter_no), test_err, 1, 'iteration number', 'error value', 'Error PLot')
        flag = input() #carry on training with more iterations
        if(flag<1):
            break
        
start = time.clock()   
main()
end = time.clock()
print 'time elapsed: ', end-start
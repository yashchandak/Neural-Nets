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


TODO:
1) convert to modular class/object based design
2) [DONE] addition of biases
3) [DONE] generalise to n number of hidden layers
4) *optimisation
        a) momentum [Done]
        b) conjugated gradient descent  
        c) regularisation
5) [DONE]normalise input and output data
6) *simulated annealing - decaying learning parameter/step size
7) [DONE] cache constant intermediate results instead of recalculating
8) end training based on difference from prv error
9) [DONE] Matrix notation for weight updates
10)*intermediate storage of weights in some file (for recovery/comparison) cPickle/CSV
11) geometric image manip
12)GPU usage
13)plotting error function and output for 2d/3d values
14) Add momentum for bias/ generalise bias (if possible)

@author: yash and ankit
"""

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import time
import features
import dataset

"""
NETWORK TOPOLOGY
"""
e           = 2.718281828
inp         = dataset.inp               #input vector dimensions:
nodes_output= dataset.output            #number of outputs
learning_rate= 0.3
momentum    = 0.3
iter_no     = 100                      #training iterations

"""
DATA generation, internediate values and weights initialisation
"""
data        = dataset.data              #get data samples
test        = dataset.test              #get test samples
err         = np.zeros(iter_no)         #keep track of sample error after each iteration
test_err    = np.zeros(iter_no)         #keep track of test error after each iteration

topology    = np.array([inp,512,512,nodes_output])
depth       = topology.size - 1

synapses    = [(np.random.random((size2,size1))-0.5)*2 for size1,size2 in zip(topology[0:depth],topology[1:depth+1])]
prv_update  = [np.zeros((size2,size1)) for size1,size2 in zip(topology[0:depth],topology[1:depth+1])]
curr_update = [np.zeros((size2,size1)) for size1,size2 in zip(topology[0:depth],topology[1:depth+1])]
bias        = [(np.random.random(size)-0.5)*2 for size in topology[1:]]
receptors   = [np.zeros(size, 'float') for size in topology[1:]] #does not have inputs
deltas      = [np.zeros(size, 'float') for size in topology[1:]]

#for Relu
#synapses    = [np.random.random((size2,size1)) for size1,size2 in zip(topology[0:depth],topology[1:depth+1])]
#bias        = [np.random.random(size) for size in topology[1:] ]

def activate(z, derivative = False, fn = 'Sigmoid' ):
    #Sigmoidal activation function    
    if fn == 'Sigmoid':
       
        if derivative:
            return z*(1-z)        
        return 1/(1+e**-z)
    
    #Relu activation function    
    elif fn == 'Relu':
        if derivative:
            return np.array([1 if item>0 else 0.01 for item in z])
        else:
            return np.array([max(0.01, item) for item in z])
            
    #tanh activation function
    elif fn == 'Tanh':
        if derivative:
            return 1-(z**2)
        else:
            return (1-e**(-2*z))/(1+e**(-2*z))
    
def plotit(x,y, fig, xlabel, ylabel, title):
    plt.figure(1)
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()    
    
    
def train_nets():
    global  err, test_err, deltas, synapses, prv_update, curr_update
    
    error = 0    
    for epoch in xrange(iter_no):        
        #update based on each data point
    
        error_sum = 0
        test_error_sum = 0
        
        for i in xrange(len(data)):
            inputs, expected = dataset.get_data(i)
            execute_net(inputs)
            error = expected - receptors[depth-1]   #error vector corresponding to each output
            #print error
            error_sum += sum(abs(error))
                     
            #backpropagation using dynamic programming
            deltas[depth-1] = activate(receptors[depth-1],True)*error
            for index in xrange(depth-2, -1, -1):
                deltas[index] = activate(receptors[index],True)*synapses[index+1].transpose().dot(deltas[index+1])
            
            #update all the weights
            for index in xrange(depth-1, 0, -1):
                curr_update[index]  = deltas[index].reshape(topology[index+1],1)*receptors[index-1]
                synapses[index]     += learning_rate*curr_update[index] + momentum*prv_update[index]
                bias[index]         += learning_rate*deltas[index]
                
            curr_update[0] = deltas[0].reshape(topology[1],1)*inputs    
            synapses[0] += learning_rate*curr_update[0] + momentum*prv_update[0]
            bias[0]     += learning_rate*deltas[0]
            
            prv_update = curr_update
         
        for i in xrange(len(test)):
            inputs, expected = dataset.get_test(i)
            execute_net(inputs)
            
            tt = np.zeros(nodes_output)
            pos = np.argmax(receptors[depth-1])
            tt[pos] = 1            
                
            test_error_sum += sum(abs(expected - tt))
            #test_error_sum += sum(abs(expected - receptors[depth-1]))
        
        err[epoch] = error_sum/len(data)
        test_err[epoch] = test_error_sum/(2*len(test)) #single misclassification creates an error sum of 2.
        
        if epoch%1 == 0:
            print "Iteration no: ", epoch, "    error: ", err[epoch], " test error: ", test_err[epoch]

    
def execute_net(inputs):
    #compute one fwd pass of the network
    global synapses, receptors

    #activate the nodes based on sum of incoming synapses    
    receptors[0] = activate(synapses[0].dot(inputs) )#+ bias[0] #activate first time based on inputs
    for index in xrange(1,depth):     
        receptors[index] = activate(synapses[index].dot(receptors[index-1]) + bias[index])
     
def predict(img):    
    execute_net(features.get_HoG(img))
    print receptors[depth-1]
    pos = np.argmax(receptors[depth-1])
    print dataset.folders[pos]

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
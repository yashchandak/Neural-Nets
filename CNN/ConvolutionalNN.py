# -*- coding: utf-8 -*-
"""
Created on Thu Feb 04 17:17:10 2016

@author: yash

ToDo:
[1] :   Discard previous activations of layers during test time (on pi)
http://cs231n.github.io/convolutional-networks/

[2] :   Check activaiton of 2d matrices
[3] : ndimage.convolve flips the filter, shouldn't be a problem though as it is flipped throughout the program
[4] : First filter works only on 2d image right now

correctness of hyper parameters :
[(layer1 - filter size + 2*padding)/stride] + 1  =  (an integer)

"""

import scipy as sp
import numpy as np
import time
import activation as act
import plotting as plot
import features
import dataset

"""------------------------ General Variables ----------------------------"""
e           = 2.718281828
iter_no     = 100                       #training iterations

inp         = dataset.inp               #input vector dimensions, should be power of 2
nodes_output= dataset.output            #number of outputs 
data        = dataset.data              #get data samples
test        = dataset.test              #get test samples

err         = np.zeros(iter_no)         #keep track of sample error after each iteration
test_err    = np.zeros(iter_no)         #keep track of test error after each iteration

"""----------------------- Variables for Conv NNet -----------------------"""
filter_size = 5
filter_count= 5

conv_layers = 3
stride      = 1
pool_size   = 2


#filters = convlayer no., filter no., 3d filter dimension (connecting all previous filters)
filters     = np.random.randn(conv_layers, filter_count, filter_count, filter_size, filter_size )  
convolved   = [np.zeros(filter_count, inp/pool_size**i, inp/pool_size**i) for i in range(conv_layers)]
conv_delta  = [np.zeros(filter_count, inp/pool_size**i, inp/pool_size**i) for i in range(conv_layers)]
switches    = [np.zeros(filter_count, inp/pool_size**i, inp/pool_size**i) for i in range(1, conv_layers+1)]
pooled      = [np.zeros(filter_count, inp/pool_size**i, inp/pool_size**i) for i in range(1, conv_layers+1)]

conv_bias   = np.zeros(conv_layers, filter_count)
 
"""---------------------- Variables for FC NNet -------------------------"""

learning_rate= 0.3
momentum    = 0.3

topology    = np.array([inp/(pool_size**(conv_layers-1)),1024,nodes_output])
depth       = topology.size - 1

synapses    = [np.random.randn(size2,size1) for size1,size2 in zip(topology[0:depth],topology[1:depth+1])]
prv_update  = [np.zeros((size2,size1)) for size1,size2 in zip(topology[0:depth],topology[1:depth+1])]
curr_update = [np.zeros((size2,size1)) for size1,size2 in zip(topology[0:depth],topology[1:depth+1])]
bias        = [np.zeros(size, 'float') for size in topology[:]]
receptors   = [np.zeros(size, 'float') for size in topology[:]] #does not have inputs
deltas      = [np.zeros(size, 'float') for size in topology[:]]


    
def train_nets():
    global  err, test_err, deltas, synapses, prv_update, curr_update
    
    error = 0    
    for epoch in xrange(iter_no):        
        #update based on each data point
    
        error_sum = 0
        test_error_sum = 0
        
        for i in xrange(len(data)):            
            receptors[0], expected = dataset.get_data(i)
            execute_net(receptors[0])
            
            error = expected - receptors[depth]   #error vector corresponding to each output
            error_sum += sum(abs(error))
                     
            """------------ backpropagation using dynamic programming ---------------"""
            
            #compute deltas for FC NNet receptors
            deltas[depth] = act.activate(receptors[depth],True)*error
            for index in xrange(depth-1, -1, -1):
                if index == 0:
                    deltas[index] = act.activate(receptors[index],True, fn='ReLu')*synapses[index].transpose().dot(deltas[index+1])
                else:                  
                    deltas[index] = act.activate(receptors[index],True)*synapses[index].transpose().dot(deltas[index+1])
            
            #update the weights of FC NNet synapses
            for index in range(depth-1, -1, -1):
                curr_update[index]  = deltas[index+1].reshape(topology[index+1],1)*receptors[index]
                synapses[index]     += learning_rate*curr_update[index] + momentum*prv_update[index]
                bias[index+1]       += learning_rate*deltas[index+1]
            
            prv_update = curr_update
        
            #compute deltas for conv NNet layers
            conv_error = deltas[0].reshape(pooled[conv_layers-1].shape) #reshape the error from FC NNet for conv NNet
            conv_delta.fill(0)                                          #flush all previous delta values
            
            for conv in range(conv_layers - 1, -1, -1):
                for fil in range(filter_count):
                    #upsampling the error matrix using switches of max pool
                    switch = switches[conv][fil]
                    for i in switch.shape[0]:
                        for j in switch.shape[1]:
                            conv_delta[conv][fil][switch[i][j][0]][switch[i][j][1]] = conv_error[fil][i][j]
        
                    #can be done outside the filter loop
                    conv_delta[conv][fil] = conv_delta[conv][fil]*act.activate(convolved[conv][fil], True ,fn = 'ReLu')
                    
        
        
        
        for i in xrange(len(test)):
            inputs, expected = dataset.get_test(i)
            execute_net(inputs)
            
            tt = np.zeros(nodes_output)
            pos = np.argmax(receptors[depth])
            tt[pos] = 1            
                
            test_error_sum += sum(abs(expected - tt))
            #test_error_sum += sum(abs(expected - receptors[depth-1]))
        
        err[epoch] = error_sum/len(data)
        test_err[epoch] = test_error_sum/(2*len(test)) #single misclassification creates an error sum of 2.
        
        if epoch%1 == 0:
            print "Iteration no: ", epoch, "    error: ", err[epoch], " test error: ", test_err[epoch]


def execute_net(inputs):
    global synapses, receptors, convolved, pooled, switches 

    #Convolutional NNet stage
    for conv in range(conv_layers):        
        for fil in range(filter_count):            
            #convolve and activate
            if conv == 0: 
                #for first convolution do it on the input image
                convolved[conv][fil] = act.activate(sp.ndimage.convolve(inputs, filters[0][fil][0], mode = 'Constant') + conv_bias[conv][fil], derivative = 'False', fn = 'ReLu')
            else:
                #do remaining colvolutions on the previous pooled matrix
                convolved[conv][fil] = act.activate(sp.ndimage.convolve(pooled[conv-1], filters[conv][fil], type = 'Constant') + conv_bias[conv][fil], derivative = 'False', fn = 'ReLu')
            
            #downsampling using max pooling
            for x in range(0,convolved[conv][fil].shape[0], pool_size):
                for y in range(0, convolved[conv][fil].shape[1], pool_size):                    
                    maximum = -999999
                    pos = (x,y)
                    
                    #selecting the max from the pooling window
                    for i in range(pool_size):
                        for j in range(pool_size):
                            
                            if convolved[conv][fil][x+i][y+j] > maximum:
                                maximum = convolved[conv][fil][x+i][y+j]
                                pos = (x+i, y+j)
       
                    pooled[conv][fil][x/pool_size][y/pool_size] = maximum
                    switches[conv][fil][x/pool_size][y/pool_size] = pos

    #pass the CNN output to FC NNet
    receptors[0] = pooled[conv_layers-1].reshape(pooled[conv_layers-1].size)
        
    for index in xrange(0,depth): 
        #activate the nodes based on sum of incoming synapses
        if index == 0:
             receptors[index+1] = act.activate(synapses[index].dot(receptors[index]))# remove bias for 0th
        else:
             receptors[index+1] = act.activate(synapses[index].dot(receptors[index]) + bias[index+1]) 

def predict(img):    
    execute_net(features.get_HoG(img))
    print receptors[depth]
    pos = np.argmax(receptors[depth])
    print dataset.folders[pos]
    
    
def main():
    while(1):
        train_nets()
        plot.plotit(range(iter_no), err, 1, 'iteration number', 'error value', 'Error PLot')
        plot.plotit(range(iter_no), test_err, 1, 'iteration number', 'error value', 'Error PLot')



start = time.clock()
main()
end = time.clock()
print 'time elapsed: ', end-start
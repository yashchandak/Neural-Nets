# -*- coding: utf-8 -*-
"""
Created on Thu Feb 04 17:17:10 2016

@author: yash

**when doing batch updates, which image do we treat as 'input' for weight updates?**
**common global variables for multiple python scripts**

ToDo:
[1] :   Discard previous activations of layers during test time (on pi) http://cs231n.github.io/convolutional-networks/
[2] : [Done]Check activaiton of 2d matrices
[3] : ndimage.convolve flips the filter, shouldn't be a problem though as it is flipped throughout the program
[4] : ***connect next level filters to all previous filter outputs, not just the one in it's axis***
[5] : conv fitler error not required to be stored for all conv layers. One is sufficient
[6] : batch updates
[7] : droupouts
[8] : momentum for filter weights
[9] : First bias problem for FC NNet
[10]: Effecient way to address array? a[1][1] or a[1,1]
[11]: Better weight initialisation
[12]: ***learing rate decay and simulated annealing***
[13]: ***Gradient checking step***
[14]: **Avg pooling**
correctness of hyper parameters :
[(layer1 - filter size + 2*padding)/stride] + 1  =  (an integer)

"""

import scipy.ndimage as sp
import numpy as np
import time
import activation as act
import plotting as plot
import dataset

"""------------------------ General Variables ----------------------------"""
e           = 2.718281828
iter_no     = 1500                       #training iterations

inp         = dataset.inp               #input vector dimensions, should be power of 2
nodes_output= dataset.output            #number of outputs 
len_data    = dataset.len_data          #get number of data samples
len_test    = dataset.len_test          #get number of test samples

err         = np.zeros(iter_no)         #keep track of sample error after each iteration
test_err    = np.zeros(iter_no)         #keep track of test error after each iteration

"""----------------------- Variables for Conv NNet -----------------------"""
learn_rate_conv = 0.5

filter_size = 11                         #each filter's dimension
step        = filter_size//2
filter_count= 8                         #number of filters per conv layer
conv_layers = 2                         #number of convolutional layers in CNN
stride      = 1                         #stride unused, as of now
pool_size   = 2                         #pooling window's dimension


#filters = convlayer no., filter no., 2d filter dimension [ToDo: 3d (connecting all previous filter results)] 
filters     = np.random.random((conv_layers, filter_count, filter_size, filter_size ))
conv_delta  = np.array([np.zeros((filter_count, inp//pool_size**i, inp//pool_size**i)) for i in range(conv_layers)])
convolved   = [np.zeros((filter_count, inp//pool_size**i, inp//pool_size**i)) for i in range(conv_layers)]
switches    = [np.zeros((filter_count, inp//pool_size**i, inp//pool_size**i, 2)) for i in range(1, conv_layers+1)]
pooled      = [np.zeros((filter_count, inp//pool_size**i, inp//pool_size**i)) for i in range(1, conv_layers+1)]

#conv_error keeps the dervative ready which just needs to be upsampled
conv_error  = [np.zeros((filter_count, inp//pool_size**i, inp//pool_size**i)) for i in range(1, conv_layers+1)]
conv_bias   = np.zeros((conv_layers, filter_count))
 
"""---------------------- Variables for FC NNet -------------------------"""

learn_rate  = 0.5
momentum    = 0.5

inp_vector  = filter_count*(inp//(pool_size**(conv_layers)))**2 #size of output from Conv NNets
topology    = np.array([inp_vector,1024,nodes_output])          #number of hidden layers and units in each
depth       = topology.size - 1

synapses    = [np.random.randn(size2,size1)/size1 for size1,size2 in zip(topology[0:depth],topology[1:depth+1])]
prv_update  = [np.zeros((size2,size1)) for size1,size2 in zip(topology[0:depth],topology[1:depth+1])]
curr_update = [np.zeros((size2,size1)) for size1,size2 in zip(topology[0:depth],topology[1:depth+1])]
bias        = [np.zeros(size, 'float') for size in topology[:]]
receptors   = [np.zeros(size, 'float') for size in topology[:]] 
deltas      = [np.zeros(size, 'float') for size in topology[:]]

"""------------------------------------------------------------------------"""
    
def upsample(conv_del, conv_err, switch):
    #upsampling the error matrix using switches of max pool
    for i in range(switch.shape[0]):
        for j in range(switch.shape[1]):
            conv_del[switch[i][j][0]][switch[i][j][1]] = conv_err[i][j]

    

def downsample(convol, pool, switch):
    #downsampling using max pooling
    for x in range(0,convol.shape[0], pool_size):
        for y in range(0, convol.shape[1], pool_size):    
            
            """use np.argmax and max for simplicity"""
            maximum = -999999
            pos = [x,y]
            
            #selecting the max from the pooling window
            for i in range(pool_size):
                for j in range(pool_size):
                    
                    if convol[x+i][y+j] > maximum:
                        maximum = convol[x+i][y+j]
                        pos = [x+i, y+j]
   
            pool[x/pool_size][y/pool_size] = maximum
            switch[x/pool_size][y/pool_size][0] = pos[0]
            switch[x/pool_size][y/pool_size][1] = pos[1]
    


def FC_fwdPass(inputs, receptors, synapses, bias):
    receptors[0] = inputs.reshape(inputs.size)
    for index in xrange(0,depth): 
        receptors[index+1] = act.activate(synapses[index].dot(receptors[index]) + bias[index+1]) 
    


def CNN_fwdPass(inputs, convolved, filters, conv_bias, pooled, switches):
    #Convolutional NNet stage
    for conv in range(conv_layers):        
        for fil in range(filter_count):            
            #convolve and activate
            cur = inputs #for first convolution do it on the input image
            if conv > 0: 
                cur = pooled[conv-1][fil]
            convolved[conv][fil] = act.activate(sp.convolve(cur, filters[conv][fil], mode = 'constant') + conv_bias[conv][fil], 'ReLu')
            downsample(convolved[conv][fil], pooled[conv][fil], switches[conv][fil])

    

def evaluate(test_err, epoch):  
    test_error_sum = 0      
    #compute the validation set error        
    for i in xrange(len_test):
        inputs, expected = dataset.get_test(i)
        execute_net(inputs)
        
        tt = np.zeros(nodes_output)
        pos = np.argmax(receptors[depth])
        tt[pos] = 1            
            
        test_error_sum += sum(abs(expected - tt))
        
    test_err[epoch] = test_error_sum/(2*len_test) #single misclassification creates an error sum of 2.
    


def FC_backprop(synapses, bias, deltas, error):
    global prv_update, curr_update
    #compute deltas for FC NNet receptors
    deltas[depth] = act.derivative(receptors[depth])*error
    for index in xrange(depth-1, -1, -1):        
        fn = 'Sigmoid'
        if index == 0: #index 0 has the ReLu output of Conv NNet
            fn = 'ReLu'  
        deltas[index] = act.derivative(receptors[index], fn)*synapses[index].transpose().dot(deltas[index+1])

    #update the weights of FC NNet synapses
    for index in range(depth-1, -1, -1):
        curr_update[index]  = deltas[index+1].reshape(topology[index+1],1)*receptors[index]
        synapses[index]     += learn_rate*curr_update[index] + momentum*prv_update[index]
        bias[index+1]       += learn_rate*deltas[index+1]
    
    prv_update = curr_update   
    


def CNN_backprop(filters, conv_bias, conv_delta, error, inputs):
    #compute deltas for conv NNet layers
            
    conv_error[conv_layers - 1] = error.reshape(pooled[conv_layers-1].shape) #reshape the error from FC NNet for conv NNet
    #flush all previous delta values
    conv_delta  = [np.zeros((filter_count, inp//pool_size**i, inp//pool_size**i)) for i in range(conv_layers)]
    for conv in range(conv_layers - 1, -1, -1):
        for fil in range(filter_count):
            upsample(conv_delta[conv][fil], conv_error[conv][fil], switches[conv][fil])

            w,h,prv = 0,0,0
            #compute the delta for current layer's filter #[can be done outside the filter loop]
            if conv > 0:
                conv_error[conv-1][fil] = sp.convolve(conv_delta[conv][fil],
                                                          filters[conv][fil].transpose(),
                                                          mode = 'constant' )*act.derivative(pooled[conv-1][fil], 'ReLu')
                w,h = pooled[conv-1][fil].shape
                prv = pooled[conv-1][fil]
                
            elif conv == 0:
                w,h = inputs.shape
                prv = inputs 
            
                
            for i in range(filter_size):
                for j in range(filter_size):
                    """it's convolution only!! but of 'valid' type in scipy.signal.convolve2d"""                            
                    filters[conv][fil][i][j] += learn_rate_conv*sum(prv[max(0, i-step):min(w, w+i-step),max(0, j-step):min(h, h+j-step)]
                                                                    *conv_delta[conv][fil][max(0, step-i):min(w, w+step-i),max(0, step-j):min(h, h+step-j)])
            conv_bias[conv][fil] += learn_rate_conv*sum(conv_delta[conv][fil])
    

def train_nets():
    global  err, test_err, deltas, synapses, bias, filters, conv_delta, conv_error, conv_bias 
     
    for epoch in xrange(iter_no):        
        #update based on each data sample
    
        error_sum = 0        
        for i in xrange(len_data):            
            inputs, expected = dataset.get_data(i)
            execute_net(inputs)
            
            error = expected - receptors[depth]   #error vector corresponding to each output
            error_sum += sum(abs(error))
                     
            FC_backprop(synapses, bias, deltas, error) 
            CNN_backprop(filters, conv_bias, conv_delta, deltas[0], inputs)
            
        evaluate(test_err, epoch)
        err[epoch] = error_sum/len_data        
        
        if epoch%10 == 0:
            print "Iteration no: ", epoch, "    error: ", err[epoch], " test error: ", test_err[epoch]


def execute_net(inputs):
    global synapses, bias, receptors, convolved, pooled, switches, conv_bias 

    CNN_fwdPass(inputs, convolved, filters, conv_bias, pooled, switches)        
    FC_fwdPass(pooled[conv_layers-1], receptors, synapses, bias)    #pass the CNN output to FC NNet


def predict(img):    
    execute_net(dataset.preprocess(img))
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
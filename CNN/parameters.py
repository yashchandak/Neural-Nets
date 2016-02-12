# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 10:32:30 2016

@author: yash
"""

"""
**when doing batch updates, which image do we treat as 'input' for weight updates?**
**gradient check for multi class output**
**understand and make notes for maths behind weight initialisation and learning rate for ReLu/Tanh**


ToDo:
[1]     : Discard previous activations of layers during test time (on pi) http://cs231n.github.io/convolutional-networks/
[2]     : [Done]Check activaiton of 2d matrices
[3]     : [Pointless]ndimage.convolve flips the filter, shouldn't be a problem though as it is flipped throughout the program
[4]*    : connect next level filters to all previous filter outputs, not just the one in it's axis
[5]     : conv fitler error not required to be stored for all conv layers. One is sufficient
[6]**   : batch updates
[7]     : droupouts
[8]     : momentum for filter weights
[9]     : [Done]First bias problem for FC NNet
[10]    : Effecient way to address array? a[1][1] or a[1,1]
[11]    : Better weight initialisation
[12]*   : learing rate decay and simulated annealing
[13]    : [Bugy] Gradient checking step
[14]*   : Avg pooling
[15]*   : list for activation fn of each layers
[16]**  : Batch Normalisation
[17]*   : AdaDelta/RMSProb momentums
[18]*   : Cython for CNN
[19]*   : OpenBlas with fastDot for quick matrix multiplication
correctness of hyper parameters :
[(layer1 - filter size + 2*padding)/stride] + 1  =  (an integer)

"""

import dataset
import numpy as np


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
learn_rate_conv = 0.9

filter_size = 5                         #each filter's dimension
step        = filter_size//2
filter_count= 8                         #number of filters per conv layer
conv_layers = 1                         #number of convolutional layers in CNN
stride      = 1                         #stride unused, as of now
pool_size   = 2                         #pooling window's dimension

act_fn_conv = ['ReLu']
#filters = convlayer no., filter no., 2d filter dimension [ToDo: 3d (connecting all previous filter results)] 
filters     = np.random.randn(filter_count, filter_size, filter_size)*np.sqrt(2.0/(filter_size**2))
update_fil  = np.zeros((conv_layers, filter_count, filter_size, filter_size ))
conv_delta  = np.array([np.zeros((filter_count, inp//pool_size**i, inp//pool_size**i)) for i in range(conv_layers)])
convolved   = [np.zeros((filter_count, inp//pool_size**i, inp//pool_size**i)) for i in range(conv_layers)]
switches    = [np.zeros((filter_count, inp//pool_size**i, inp//pool_size**i, 2)) for i in range(1, conv_layers+1)]
pooled      = [np.zeros((filter_count, inp//pool_size**i, inp//pool_size**i)) for i in range(1, conv_layers+1)]

#conv_error keeps the dervative ready which just needs to be upsampled
conv_error  = [np.zeros((filter_count, inp//pool_size**i, inp//pool_size**i)) for i in range(1, conv_layers+1)]
conv_bias   = np.random.random((conv_layers, filter_count))*0.1
 
 
 
 
"""---------------------- Variables for FC NNet -------------------------"""

learn_rate  = 0.5
momentum    = 0.5

inp_vector  = filter_count*(inp//(pool_size**(conv_layers)))**2 #size of output from Conv NNets
topology    = np.array([inp_vector,2048,512,nodes_output])          #number of hidden layers and units in each
act_fn      = ['ReLu', 'Sigmoid', 'Sigmoid', 'Sigmoid']
depth       = topology.size - 1

synapses    = [np.random.randn(size2,size1)*(1.0/np.sqrt(size1)) for size1,size2 in zip(topology[0:depth],topology[1:depth+1])]
prv_update  = [np.zeros((size2,size1)) for size1,size2 in zip(topology[0:depth],topology[1:depth+1])]
curr_update = [np.zeros((size2,size1)) for size1,size2 in zip(topology[0:depth],topology[1:depth+1])]
bias        = [np.random.random((size))*0.1 for size in topology[:]]
receptors   = [np.zeros(size, 'float') for size in topology[:]] 
deltas      = [np.zeros(size, 'float') for size in topology[:]]

"""------------------------------------------------------------------------"""
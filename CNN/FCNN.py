# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 11:30:39 2016

@author: yash
"""
import parameters as net
import activation as act

def fwdPass(inputs, receptors, synapses, bias):
    receptors[0] = inputs.reshape(inputs.size)
    for index in xrange(0,net.depth): 
        receptors[index+1] = act.activate(synapses[index].dot(receptors[index]) + bias[index+1]) 
 

       
def backprop(receptors, synapses, bias, deltas, error):
    #compute deltas for FC NNet receptors
    #receptors[0] = inputs.reshape(inputs.size)
    deltas[net.depth] = act.derivative(receptors[net.depth])*error
    for index in xrange(net.depth-1, -1, -1):        
        fn = 'Sigmoid'
        if index == 0: #index 0 has the ReLu output of Conv NNet
            fn = 'ReLu'  
        deltas[index] = act.derivative(receptors[index], fn)*synapses[index].transpose().dot(deltas[index+1])

    #update the weights of FC NNet synapses
    for index in range(net.depth-1, -1, -1):
        net.curr_update[index]  = deltas[index+1].reshape(net.topology[index+1],1)*receptors[index]
        synapses[index]     += net.learn_rate*net.curr_update[index] + net.momentum*net.prv_update[index]
        bias[index+1]       += net.learn_rate*deltas[index+1]
    
    net.prv_update = net.curr_update   
    
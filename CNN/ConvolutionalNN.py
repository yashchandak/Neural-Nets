# -*- coding: utf-8 -*-
"""
Created on Thu Feb 04 17:17:10 2016

@author: yash
"""
import parameters as net
import FCNN as fcnn
import CNN as cnn
import numpy as np
import time
import plotting as plot
import dataset

def predict(img):    
    execute_net(dataset.preprocess(img))
    print net.receptors[net.depth]
    pos = np.argmax(net.receptors[net.depth])
    print dataset.folders[pos]

    

def evaluate(receptors, test_err, epoch):  
    test_error_sum = 0      
    #compute the validation set error        
    for i in xrange(net.len_test):
        inputs, expected = dataset.get_test(i)
        execute_net(inputs)
        
        tt = np.zeros(net.nodes_output)
        pos = np.argmax(receptors[net.depth])
        tt[pos] = 1            
            
        test_error_sum += sum(abs(expected - tt))
        
    test_err[epoch] = test_error_sum/(2*net.len_test) #single misclassification creates an error sum of 2.
    

def gradient_check(inputs, expected):
    #use ONLY to check the backprop implementation with one class output
    
    
    #randomly check for any one of the filter weights
    c,f,i,j = net.filters.shape
    c = np.random.random_integers(0,c-1)
    f = np.random.random_integers(0,f-1)
    i = np.random.random_integers(0,i-1)
    j = np.random.random_integers(0,j-1)
    
    got_grad = net.update_fil[c][f][i][j] 
    
    cur = net.filters[c][f][i][j]
    prv = cur - net.learn_rate_conv*net.update_fil[c][f][i][j]
    
    h = 0.00001
    net.filters[c][f][i][j] = prv - h
    execute_net(inputs)    
    error1 = 0.5*(expected - net.receptors[net.depth])**2
    
    net.filters[c][f][i][j] = prv + h 
    execute_net(inputs)
    error2 = 0.5*(expected - net.receptors[net.depth])**2
    
    true_grad = (error2[0] - error1[0])/(2*h)
    
    if abs(true_grad - got_grad) > 0.00001:
        print "Gradient calculation is wrong!"
        print "expected Gradient: ", true_grad, "  Received Grad: ", got_grad
        
    net.filters[c][f][i][j] = cur #restore the updated value
    
    
def train_nets():
    for epoch in xrange(net.iter_no):        
        #update based on each data sample    
        error_sum = 0        
        for i in xrange(net.len_data):            
            inputs, expected = dataset.get_data(i)
            
            execute_net(inputs)
            
            error = expected - net.receptors[net.depth]   #error vector corresponding to each output
            error_sum += sum(abs(error))
             
            learn(inputs, error) 
            
            #check if learning algorithm is working properly
            gradient_check(inputs, expected)
            
        evaluate(net.test_err, epoch)
        net.err[epoch] = error_sum/net.len_data        
        
        if epoch%10 == 0:
            print "Iteration no: ", epoch, "    error: ", net.err[epoch], " test error: ", net.test_err[epoch]



def execute_net(inputs):
    cnn.fwdPass(inputs, net.convolved, net.filters, net.conv_bias, net.pooled, net.switches)        
    fcnn.fwdPass(net.pooled[net.conv_layers-1], net.receptors, net.synapses, net.bias)    #pass the CNN output to FC NNet


def learn(inputs, error):    
    fcnn.backprop(net.synapses, net.bias, net.deltas, error) 
    cnn.backprop(net.filters, net.conv_bias, net.pooled, net.switches, net.conv_error, net.conv_delta, net.deltas[0], inputs)
    
    
def main():
    while(1):
        train_nets()
        plot.plotit(range(net.iter_no), net.err, 1, 'iteration number', 'error value', 'Error PLot')
        plot.plotit(range(net.iter_no), net.test_err, 1, 'iteration number', 'error value', 'Error PLot')



start = time.clock()
main()
end = time.clock()
print 'time elapsed: ', end-start
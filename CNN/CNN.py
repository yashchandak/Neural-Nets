# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 11:33:16 2016

@author: yash
"""
import parameters as net
import activation as act
import pooling as pool
import scipy.ndimage as sp
import numpy as np


def fwdPass(inputs, convolved, filters, conv_bias, pooled, switches):
    #Convolutional NNet stage
    for conv in range(net.conv_layers):        
        for fil in range(net.filter_count):            
            #convolve and activate          
            if conv == 0:
                #for first convolution do it on the input image
                convolved[conv][fil] = act.activate(sp.convolve(inputs, filters[conv][fil][0], mode = 'constant') + conv_bias[conv][fil], net.act_fn_conv[conv])
                
            elif conv > 0: 
                temp = np.zeros(convolved[conv][fil].shape)
                for prv_fil in range(net.filter_count):
                    temp += sp.convolve(pooled[conv-1][prv_fil], filters[conv][fil][prv_fil], mode = 'constant')
                convolved[conv][fil] = act.activate(temp + conv_bias[conv][fil], net.act_fn_conv[conv])
            
            pool.downsample(convolved[conv][fil], pooled[conv][fil], switches[conv][fil])



def backprop(filters, conv_bias, pooled, switches, conv_error, conv_delta, error, inputs):
    #compute deltas for conv NNet layers            
    conv_error[net.conv_layers - 1] = error.reshape(pooled[net.conv_layers-1].shape) #reshape the error from FC NNet for conv NNet
    #flush all previous delta values
    conv_delta  = [np.zeros((net.filter_count, net.inp//net.pool_size**i, net.inp//net.pool_size**i)) for i in range(net.conv_layers)]

    for conv in range(net.conv_layers - 1, -1, -1):
        for fil in range(net.filter_count):
            
            pool.upsample(conv_delta[conv][fil], conv_error[conv][fil], switches[conv][fil])
            w,h,prv = 0,0,0
            #compute the delta for current layer's filter #[can be done outside the filter loop]'
            #derivative for prv convolved layer can be computed once only and stored
            if conv > 0:
                for prv_fil in range(net.filter_count):
                    conv_error[conv-1][prv_fil] += sp.convolve(conv_delta[conv][fil],
                                                          filters[conv][fil][prv_fil].transpose(),
                                                          mode = 'constant' )*act.derivative(pooled[conv-1][prv_fil], net.act_fn_conv[conv-1])

                    #for prv_fil in range(net.filter_count):
                    w,h = pooled[conv-1][prv_fil].shape
                    prv = pooled[conv-1][prv_fil]
                    for i in range(net.filter_size):
                        for j in range(net.filter_size):
                            net.update_fil[conv][fil][prv_fil][i][j] =  sum(sum(prv[max(0, i-net.step):min(w, w+i-net.step),max(0, j-net.step):min(h, h+j-net.step)]*conv_delta[conv][fil][max(0, net.step-i):min(w, w+net.step-i),max(0, net.step-j):min(h, h+net.step-j)]))
                        
            elif conv == 0:
                w,h = inputs.shape
                prv = inputs 
                for i in range(net.filter_size):
                    for j in range(net.filter_size):
                        """it's convolution only!! but of 'valid' type in scipy.signal.convolve2d""" 
                        net.update_fil[conv][fil][0][i][j] =  sum(sum(prv[max(0, i-net.step):min(w, w+i-net.step),max(0, j-net.step):min(h, h+j-net.step)]*conv_delta[conv][fil][max(0, net.step-i):min(w, w+net.step-i),max(0, net.step-j):min(h, h+net.step-j)]))
                        

            filters[conv][fil]   += net.learn_rate_conv*net.update_fil[conv][fil]
            conv_bias[conv][fil] += net.learn_rate_conv*sum(sum(conv_delta[conv][fil]))
            
        conv_error[conv].fill(0) #reset conv_error for each layer     
            
            
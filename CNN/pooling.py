# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 11:30:37 2016

@author: yash
"""
import parameters as net
import numpy as np

def upsample(conv_del, conv_err, switch):
    #upsampling the error matrix using switches of max pool
    
    for i in range(switch.shape[0]):
        for j in range(switch.shape[1]):
            conv_del[switch[i][j][0]][switch[i][j][1]] = conv_err[i][j]

    

def downsample(convol, pool, switch):
    #downsampling using max pooling
    for x in range(0,convol.shape[0], net.pool_size):
        for y in range(0, convol.shape[1], net.pool_size):    
            
            """use np.argmax and max for simplicity"""
            maximum = -999999
            pos = [x,y]
            
            #selecting the max from the pooling window
            for i in range(net.pool_size):
                for j in range(net.pool_size):
                    
                    if convol[x+i][y+j] > maximum:
                        maximum = convol[x+i][y+j]
                        pos = [x+i, y+j]
   
            pool[x/net.pool_size][y/net.pool_size] = maximum
            switch[x/net.pool_size][y/net.pool_size][0] = pos[0]
            switch[x/net.pool_size][y/net.pool_size][1] = pos[1]
    
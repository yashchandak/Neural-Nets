# -*- coding: utf-8 -*-
"""
Created on Mon Feb 08 13:33:01 2016

ToDo:

[1] : Improve activation functions for multidimensional arrays


@author: yash
"""
import numpy as np

e = 2.718281828

def activate(z, derivative = False, fn = 'Sigmoid' ):
    
    #Sigmoidal activation function
    if fn == 'Sigmoid':
       
        if derivative:
            return z*(1-z)   
        else:
            return 1/(1+e**-z)
    
    #Relu activation function    
    elif fn == 'ReLu':
        if len(z.shape) == 1:
            if derivative:
                return np.array([1 if item>0.01 else 0.01 for item in z])
            else:
                return np.array([max(0.01, item) for item in z])
        
        elif len(z.shape) == 2:
            w,h = z.shape
            if derivative:
                return np.array([[1 if item>0.01 else 0.01 for item in z[i]] for i in range(h)])
            else:
                return np.array([[max(0.01, item) for item in z[i]] for i in range(h)])            
            
    #tanh activation function
    elif fn == 'Tanh':
        if derivative:
            return 1-(z**2)
        else:
            return (1-e**(-2*z))/(1+e**(-2*z))
            
    else:
        print 'ERROR! invalid function!'
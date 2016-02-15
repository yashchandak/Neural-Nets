# -*- coding: utf-8 -*-
"""
Created on Mon Feb 08 13:33:01 2016

ToDo:

[1] : Improve activation functions for multidimensional arrays


@author: yash
"""
import numpy as np

e = 2.718281828



def activate(z, fn = 'Sigmoid' ):
    #Sigmoidal activation function
    if fn == 'Sigmoid':
        return 1/(1+e**-z)
    
    #Relu activation function    
    elif fn == 'ReLu':
        if len(z.shape) == 1:
            return np.array([max(0.01, item) for item in z])
        
        elif len(z.shape) == 2:
            w,h = z.shape
            return np.array([[max(0.01, item) for item in z[i]] for i in range(h)])
        else:
            print 'Error! Relu activation not defined for this shape: ' , z.shape
            
    #tanh activation function
    elif fn == 'Tanh':
            return (1-e**(-2*z))/(1+e**(-2*z))
            
    else:
        print 'ERROR! invalid function!'




        
def derivative(z, fn = 'Sigmoid'):
    #Sigmoidal derivative function
    if fn == 'Sigmoid' or fn == 'ReLu':
            return z*(1-z)   
    
    #Relu derivative function    
    elif fn == 'ReLu':
        if len(z.shape) == 1:
            return np.array([1 if item>0.01 else 0.01 for item in z])
        
        elif len(z.shape) == 2:
            w,h = z.shape
            return np.array([[1 if item>0.01 else 0.01 for item in z[i]] for i in range(h)])  
        else:
            print 'Error! Relu derivative not defined for this shape: ' , z.shape
            
    #tanh derivative function
    elif fn == 'Tanh':
            return 1-(z**2)
            
    else:
        print 'ERROR! invalid function!'
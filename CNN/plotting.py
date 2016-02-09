# -*- coding: utf-8 -*-
"""
Created on Mon Feb 08 13:51:00 2016

@author: yash
"""
import matplotlib.pyplot as plt

def plotit(x,y, fig, xlabel, ylabel, title):
    plt.figure(1)
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()    
    
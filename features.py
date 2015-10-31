# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 19:43:46 2015

@author: yash and ankit 
"""
import cv2
import numpy as np
import math

dim = 32
inp = 64
window = 4
bins = 8


def get_HoG(img):
    
    #resize to dim + 1 to accomodate for dx and dy
    
    img = cv2.resize(img, (dim+1, dim+1))
    img = img.astype(int)
    features = np.zeros(inp)
    hist = np.zeros(bins)
    index, dx, dy, mag, ang, pos = 0,0,0,0,0,0
    div = 6.28/bins 
    
    #print dim, inp, window, bins
    
    for r in xrange(0,dim, window):
        for c in xrange(0,dim, window):
            hist.fill(0) #reset histogram bins
            
            #calculate HoG of the subWindow
            for i in xrange(window):
                for j in xrange(window):
                    dy = img[r+i+1][c+j] - img[r+i][c+j]
                    dx = img[r+i][c+j+1] - img[r+i][c+j]
                    mag = dx**2 + dy**2
                    ang = math.atan2(dy,dx) + 3.13 #range = 0 - 6.27
                    pos = int(ang/div)
                    hist[pos] += mag
            
            #the feature from this window represents highest valued direction
            features[index] = np.argmax(hist)*1.0/(bins-1)
            #print 'hist: ', hist, np.argmax(hist)*1.0/(bins-1)
            index +=1
    #print "feat innnn features: ",features        
    return features
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 19:43:46 2015

@author: yash and ankit 
"""
import cv2
import numpy as np
import math

dim = 32
window = 4
bins = 8
#inp = 64
inp = ((dim/window)**2)*bins

def get_HoG(img):
    
    #resize to dim + 1 to accomodate for dx and dy    
    img = cv2.resize(img, (dim+1, dim+1))
    img = img.astype(int)
    features = np.zeros(inp)
    features2 = np.zeros(inp)
    hist = np.zeros(bins)
    index, dx, dy, mag, ang, pos = 0,0,0,0,0,0
    div = 6.28/bins 
    count = 0
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
            
            #vector of 1 and 0 for gradient direction
            features2[count*bins + np.argmax(hist)] = 1
            count += 1

             #suppressing regions without dominant gradient
#            highest = np.argmax(hist)
#            val = hist[highest]
#            hist[highest] = 0
#            highest2 = np.argmax(hist)
#            
#            
#            if val > 1.25 * hist[highest2]:
#                features[index] = (highest+1)*1.0/bins
#            else:
#                features[index] = 0  
             
            #the feature from this window represents highest valued direction 
            #features[index] = np.argmax(hist)*1.0/(bins-1)          #std. feature
            #features[index] = np.argmax(hist)*1.0/(bins-1) - 0.5    #centered around 0
            index +=1
           
    return features2
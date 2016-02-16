# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 14:05:12 2016

@author: yash

TODO:

[1] :   have to create all the named folders otherwise imwrite doesnt work

"""

import numpy as np
import os, os.path
import cv2

valid_images = [".jpg",".gif",".png",".tga", ".pgm"]
path = "D:/ToDo/DRDO/IRDatabase/Data/"
folders = ['Tracked']#,'watch','Leopards','butterfly','starfish','scorpion','revolver']
write_path = "D:/ToDo/DRDO/IRDatabase/Data/manipulated/"

#    
#def warped():
#    morphed = []
#    write(morphed, )
#
#
def inverted(img, s):
    img = img.astype(float)
    im = cv2.invert(img)
    write(im, 'inverted/inv_'+s+'.jpg' )
    
def flipped(img, s):
    
    im = cv2.flip(img, 1)
    write(im, 'flipped/flip_'+s+'.jpg' )
        
#def rotated():
#    
#def blurred():
#    
#def noisy():
#    
#def occluded():
#    
#def contrasted():
#    


def read_from_folder(path):
    imgs = []
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        filename = path+'/'+f
        img = cv2.imread(filename,0)
        #feat = preprocess(img)
        imgs.append(img)
    return imgs



def compile_data():
    global data,test
    
    for idx, folder in enumerate(folders):
        imgs = read_from_folder(path+folder)
        
        for i in range(len(imgs)):
            #cv2.imshow('window', imgs[i])
            s = str(i)
#            warped(imgs[i],s)
            inverted(imgs[i],s)
#            flipped(imgs[i],s)
#            rotated(imgs[i],s)
#            blurred(imgs[i],s)
#            noisy(imgs[i],s)
#            occluded(imgs[i],s)
#            contrasted(imgs[i],s)
    
def write(img,  tag):
    add = write_path + tag
    cv2.imwrite(add, img)

compile_data()

print 'Datset made successfully'
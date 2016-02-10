# -*- coding: utf-8 -*-
"""
Created on Mon Feb 08 14:07:41 2016

@author: yash

TODO
1) use CIFAR 10

"""
import numpy as np
import os, os.path
import cv2


dim = 32
inp = dim
valid_images = [".jpg",".gif",".png",".tga", ".pgm"]
path = "D:/ToDo/datasets/101_ObjectCategories/"
folders = ['airplanes','car_UIUC','Motorbikes','Faces_easy']#,'watch','Leopards','butterfly','starfish','scorpion','revolver']
output = len(folders)
data = []
test = []

def preprocess(img):
    return cv2.resize(img, (dim, dim))

def get_data(index):
    return data[index][0], data[index][1]

def get_test(index):
    return test[index][0], test[index][1]    
    
def read_from_folder(path, val):
    imgs = []
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        filename = path+'/'+f
        img = cv2.imread(filename,0)
        feat = preprocess(img)
        imgs.append([feat, val])
    return imgs

def compile_data():
    global data,test
    
    for idx, folder in enumerate(folders):
        category = np.zeros(output)
        category[idx] = 1
        imgs = read_from_folder(path+folder, category)
        count = int(0.8*len(imgs))

        np.random.shuffle(imgs)        
        
        data.extend(imgs[:count])
        test.extend(imgs[count:])
    
    np.random.shuffle(data)
    #np.random.shuffle(test)
    

compile_data()
print 'Datset made successfully'
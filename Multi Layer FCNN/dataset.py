# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:41:03 2015

@author: yash and ankit

TODO
1) use CIFAR 10

"""
import numpy as np
import os, os.path
import cv2
import features

inp = 512#64
dim = 32
valid_images = [".jpg",".gif",".png",".tga", ".pgm"]
path = "D:/ToDo/datasets/101_ObjectCategories/"
folders = ['airplanes','car_UIUC','Motorbikes','Faces_easy',]#'watch','Leopards','butterfly','starfish','scorpion','revolver']
output = len(folders)
data = []
test = []

#data2 = [[0,0,0,0,0],[0,0,0,1,1],[0,0,1,0,1],[0,0,1,1,0],[0,1,0,0,1],[0,1,0,1,0],[0,1,1,0,0],[0,1,1,1,1],[1,0,0,0,1],[1,0,0,1,0],[1,0,1,0,0],[1,0,1,1,1],[1,1,0,0,0],[1,1,0,1,1],[1,1,1,0,1],[1,1,1,1,0]]

def generate_data():
    global data, test
    data = [[0,0,0,0,0],[0,0,0,1,1],[0,0,1,0,1],[0,0,1,1,0],[0,1,0,0,1],[0,1,0,1,0],[0,1,1,0,0],[0,1,1,1,1],[1,0,0,0,1],[1,0,0,1,0],[1,0,1,0,0],[1,0,1,1,1],[1,1,0,0,0],[1,1,0,1,1],[1,1,1,0,1],[1,1,1,1,0]]
    test = data    
    #data = [[.1,.1,0.75],[.4,.1,0.75],[.1,.4,0.75],[.1,.7,0],[.1,.9,0], [.3,.8,0], [.7,.7,0.25],[.6,.99,0.25],[.9,.7,0.25],[.8,.1,1],[.7,.3,1],[.99,.2,1],[.5,.5,0.5],[.8,.5,0.5], [.4,.7,0.5], [.6,.8,0.5]]
    #data = [[.1,.1,0,0],[.4,.1,0,0],[.1,.4,0,0],[.1,.7,0,1],[.1,.9,0,1], [.3,.8,0,1], [.7,.7,1,0],[.6,.99,1,0],[.9,.7,1,0],[.8,.1,1,1],[.7,.3,1,1],[.99,.2,1,1]]
    #data = [[0,0,0],[0,1,1],[1,0,1],[1,1,0]] #XOR   
    #data = [[0,0,0],[0,1,1],[1,0,1],[1,1,0]] #AND
    #data = np.array([[0,0],[1,1],[4,2],[5,4],[6,6],[7,7],[8,9],[9,9],[12,10],[13,9],[15,8],[16,7],[20,4],[21,2],[23,0],[24,0]], dtype = 'float')
    #data = np.array([[.1,.2,.3,.1],[.3,.5,.8,.2]])    #manual test data
    
    #return data

def get_data(index):
    #in1, in2, out1 = data[index]             #parse the data to get input and expected output
    #in1, out1 = data[index]    
#    in1, in2, in3, in4, out1 = data[index]
#    return np.array([in1,in2, in3, in4]),np.array([out1])
    return data[index][0], data[index][1]



def get_test(index):
#    in1, in2, in3, in4, out1 = test[index]
#    return np.array([in1,in2, in3, in4]),np.array([out1])
    return test[index][0], test[index][1]    


    
def read_from_folder(path, val):
    imgs = []
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        filename = path+'/'+f
        img = cv2.imread(filename,0)
        feat = features.get_HoG(img)
        imgs.append([feat, val])
    return imgs
    


def set_feature_parameters():
    features.bins = 8       #number of bins in HoG
    features.inp = inp       #dimension of input vector for neural network
    features.window = 4     #window size = local region for histogram calculation
    features.dim = dim       #image resized dimension
    
    

def compile_data():
    global data,test
    set_feature_parameters()  
    
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
#generate_data()
print 'Datset made successfully'
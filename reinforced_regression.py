# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 18:25:06 2015

Regression Reinforcement learning

TODO:

remove numpy dependency
single module


@author: yash
"""
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

""" format = [ temp, value] ENTER MAX AND MIN VALUE THAT CAN EVER BE PROVIDED"""
data = np.array([[0,0],[1,1],[4,2],[5,4],[6,6],[7,7],[8,9],[9,9],[12,10],[13,9],[15,8],[16,7],[20,4],[21,2],[23,0],[24,0]], dtype = 'float')
#data = np.array([[0,0],[10,1],[15,4],[25,7],[27,8],[30,9],[35,9],[40,9],[45,10],[50,10],[60,10],[70,10]], dtype = 'float')
#data = np.array([[0,0],[1,1],[2,8],[3,27],[4,64],[5,125],[6,216],[7,343],[8,512],[9,729],[10,1000]],dtype = 'float')
#data[:,1] = data[:,1] + 1000

#init plot of given data
plt.plot(data[:,0],data[:,1],'b+')

max_x = np.amax(data[:,0])
max_y = np.amax(data[:,1])
min_x = np.amin(data[:,0])
min_y = np.amin(data[:,1])

#normalise the data
data[:,0] = (data[:,0]-min_x)/(max_x-min_x)
data[:,1] = (data[:,1]-min_y)/(max_y-min_y)

plt.xlabel('temp')
plt.ylabel('value')
plt.title('Fan speed')
plt.show()

plots = 10
degree = 3
delta = 9999

t = np.array(rand(degree+1), dtype = 'float')
weights = np.ones(len(data))

print 'initial coeff: ',t


def plot():
    #normalize the input
    xx = (np.array(range(int(min_x)-10 ,int(max_x)+20), dtype = 'float') - min_x)/(max_x - min_x)
    yy = np.array([],dtype = 'float')
    for x in xx:
        feats = np.array([pow(x,i) for i in range(degree+1)], dtype = 'float')         
        y = t.dot(feats)
        yy = np.append(yy,y)
        
    #de-normalize and display   
    plt.plot(xx*(max_x - min_x) + min_x, yy*(max_y - min_y) + min_y)
    plt.show()
    
    
#fig = plt.figure()
def learn(Training_Iter = 10000, learning_rate = 0.5):
    global t, weights, data
    #while(np.sum(abs(delta)) > 0.01):
    for i in range(Training_Iter):
        #prv_t = t.copy()  
        #display 'plots' intermediate plots    
        if i%(Training_Iter/plots)== 0 :
            plot()
            
        for index in range(len(data)):
            #print val[0], ' ',val[1]
            x = data[index][0]        
            #array of features i.e 1, x, x^2, x^3 ...
            feats = np.array([pow(x,i) for i in range(degree+1)], dtype = 'float')        
            #calculate dot product        
            predicted = t.dot(feats)        
            #SGD, error in prediction [differentiated]
            err = weights[index]*(data[index][1]-predicted)        
            #Update the co-effs
            t = t + learning_rate*err*feats   
            
        #delta = t - prv_t
        #print '\n\n',t,'\n', prv_t,'\n', delta, np.sum(abs(delta))
    
    print 'final co-effs: ',t
    #print delta, np.sum(abs(delta))


def user_update(decay = 0.85):
    #Dynamic updates from user
    global data, weights
    while(1):    
        
        x = input('enter param: ')
        if x == -999:
            break    
        x = (x - min_x)/(max_x - min_x)  
        
        #make prediction
        feats = np.array([pow(x,i) for i in range(degree+1)], dtype = 'float')        
        predicted = t.dot(feats) 
        print 'predicted: ', predicted*(max_y - min_y) + min_y
        
        #get correction        
        desired = input('desired setting: ')
        if desired == -999:
            continue
        desired = (desired - min_y)/(max_y - min_y)
        
        #decay previous values
        weights *= decay
        
        #add new data and its weight
        weights = np.append(weights, 1)
        data = np.append(data,[[x,desired]],axis=0)
        
        #re-learn
        learn(Training_Iter = 1000, learning_rate = 0.5)
        print 'updated co-eff: ',t
        
        plot()

def main():
    learn()
    fig2 = plt.figure()
    user_update()

main()



"""NON WEIGHTED USER UPDATES

    x = (inp - min_x)/(max_x - min_x)    
    #array of features i.e 1, x, x^2, x^3 ...
    feats = np.array([pow(x,i) for i in range(degree+1)], dtype = 'float')        
    #calculate dot product        
    predicted = t.dot(feats)      
    
    print 'predicted: ', predicted*(max_y - min_y) + min_y
    desired = input('desired setting: ')
    desired = (desired - min_y)/(max_y - min_y)
        
    #SGD, error in prediction [differentiated]
    err = desired - predicted        
    #Update the co-effs
    t = t + learning_rate*err*feats  
    print 'updated co-eff: ',t
    
"""


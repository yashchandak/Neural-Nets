# -*- coding: utf-8 -*-
"""
Created on Sun Nov 01 12:13:30 2015

@author: yash and ankit
"""
    

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import time
import dataset

"""
NETWORK TOPOLOGY
"""

chroms = 30
pooling_factor = .1             #fraction of least fit chromosomes to be removed after each iteration
mutate_factor = .5
e = 2.718281828
inp = dataset.inp                 #input vector dimensions:
nodes_output  = dataset.output  #number of outputs
learning_rate = 0.5
momentum = 0.3
iter_no = 25000              #training iterations

"""
DATA generation, internediate values and weights initialisation
"""
data = dataset.data                                      #get data
test = dataset.test
chromosomes = np.zeros(chroms)
fitness = np.zeros(chroms)

err = np.zeros(iter_no)
test_err = np.zeros(iter_no)

topology = np.array([inp,32,nodes_output])
depth = topology.size - 1

receptors = [np.zeros(size, 'float') for size in topology[1:]] #does not have inputs  

def activate(z, derivative = False):
    #Sigmoidal activation function
    if derivative:
        return z*(1-z)        
    return 1/(1+e**-z)
    
def plotit(x,y, fig, xlabel, ylabel, title):
    plt.figure(1)
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()    
    
    
def generate_chromosomes():  
    global chromosomes
    
    for idx in xrange(chroms):
        synapses = [np.random.random((size2,size1)) for size1,size2 in zip(topology[0:depth],topology[1:depth+1])]
        chromosomes[idx] = synapses

    
def eval_fitness(synapses):
    error_sum = 0
    for i in xrange(len(data)):
        inputs, expected = dataset.get_data(i)
        result = execute_net(inputs, synapses)
        error = expected - result   #error vector corresponding to each output
        error_sum += sum(abs(error))
        
    return error_sum/len(data)


def test_eval(synapse):
    test_error_sum = 0
    for i in xrange(len(test)):
        inputs, expected = dataset.get_test(i)
        result = execute_net(inputs, synapse)
        
        tt = np.zeros(nodes_output)
        pos = np.argmax(result)
        tt[pos] = 1            
            
        test_error_sum += sum(abs(expected - tt))
        
    #Fitness is inversely proportional to error
    return 10.0/(0.1+test_error_sum/len(test))

    
def pool():
    global chromosomes, fitness
    
    pos_max = np.argmax(fitness)
    for idx in xrange(int(pooling_factor*chroms)):
        pos_min = np.argmin(fitness)
        
        #replace the least fit chromosome with the best one
        chromosomes[pos_min] = chromosomes[pos_max].copy()
        fitness[pos_min] = fitness[pos_max]

        
def mutate():
    global chromosomes, fitness
    
    max_fitness = np.argmax(fitness)    
    for idx in xrange(chrom):
        eps = 1/(fitness[idx]/sum(fitness))
        for i in xrange(len(synapse)):
            
    
    return 0  
    
    
def crossover():
    global chromosomes, fitness
    
    return 0
 
   
 


    
def train_nets():
    global  err, test_err, chromosomes, fitness
    
    error = 0    
    for epoch in xrange(iter_no):       
                          
        for idx in xrange(chroms):             
            fitness[idx] = eval_fitness(chromosomes[idx])
        
        best = np.argmax(fitness)
        err[epoch] = np.max(fitness)  
        test_err[epoch] = test_eval(chromosomes[best])
        
        pool()
        mutate()   
        crossover()                     
        
        if epoch%100 == 0:
            print "Iteration no: ", epoch, "    error: ", err[epoch], " test error: ", test_err[epoch]

    
def execute_net(inputs, synapses):
    global receptors

    #activate the nodes based on sum of incoming synapses    
    receptors[0] = activate(synapses[0].dot(inputs)) #activate first time based on inputs
    for index in xrange(1,depth):        
        receptors[index] = activate(synapses[index].dot(receptors[index-1]))
        
    return receptors[depth-1]
     


def main():
    while(1):
        train_nets()
        plotit(range(iter_no), err, 1, 'iteration number', 'error value', 'Error PLot')
        plotit(range(iter_no), test_err, 1, 'iteration number', 'error value', 'Error PLot')
        flag = input() #carry on training with more iterations
        if(flag<1):
            break
        
start = time.clock()   
main()
end = time.clock()
print 'time elapsed: ', end-start
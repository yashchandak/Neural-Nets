# -*- coding: utf-8 -*-
"""
Created on Sun Nov 01 12:13:30 2015

@author: yash and ankit
"""
    
#keep fitness in range(0,1)

from random import randint
import matplotlib.pyplot as plt
import numpy as np
import time
import dataset

"""
NETWORK TOPOLOGY
"""

chroms = 50
pooling_factor = .2             #fraction of least fit chromosomes to be removed after each iteration
mutate_factor = .5              #governs both number of mutation and amt of mutation
e = 2.718281828
inp = dataset.inp                 #input vector dimensions:
nodes_output  = dataset.output  #number of outputs
learning_rate = 0.5
momentum = 0.3
iter_no = 1000             #training iterations

"""
DATA generation, internediate values and weights initialisation
"""
data = dataset.data                                      #get data
test = dataset.test

chromosomes = []
fitness = np.zeros(chroms)

err = np.zeros(iter_no)
test_err = np.zeros(iter_no)

topology = np.array([inp,2,2,nodes_output])
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
        synapses = np.array([(np.random.random((size2,size1))-0.5)*5 for size1,size2 in zip(topology[0:depth],topology[1:depth+1])])
        chromosomes.append(synapses)
    chromosomes=np.array(chromosomes)
    
        

def get_fitness(error):
    return 1.0-error
    

def train_eval(synapses):
    error_sum = 0
    for i in xrange(len(data)):
        inputs, expected = dataset.get_data(i)
        result = execute_net(inputs, synapses)
        error_sum += sum(abs(expected - result))
        
    return get_fitness(error_sum/(2*len(data)))


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
    return get_fitness(test_error_sum/(2*len(test)))

    
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
    wts = np.product(topology) #number of weights in the network
    
    for idx in xrange(chroms):
        eps = 1 - fitness[idx]
        count = mutate_factor*eps*wts #number of weights to be mutated
        for i in xrange(int(count)):
            layer = randint(0,len(chromosomes[idx])-1)
            row = randint(0,len(chromosomes[idx][layer])-1)
            col = randint(0,len(chromosomes[idx][layer][row])-1)
            
            chromosomes[idx][layer][row][col] += (np.random.rand()-0.5)*2*mutate_factor*chromosomes[idx][layer][row][col]
            
       
def crossover():
    global chromosomes, fitness    
    
    for idx in range(chroms/2):
        chrom1 = randint(0,chroms-1)
        chrom2 = randint(0,chroms-1)
        
        for layer in xrange(len(topology)-1):
            
            row,col = chromosomes[chrom1][layer].shape 
            #numpy's randint : random integers from low (inclusive) to high (exclusive)
            x1, x2 = np.random.randint(0,row,2)
            y1, y2 = np.random.randint(0,col,2)
            
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
                
            chromosomes[chrom1][layer][x1:x2][y1:y2], chromosomes[chrom2][layer][x1:x2][y1:y2] = chromosomes[chrom2][layer][x1:x2][y1:y2], chromosomes[chrom1][layer][x1:x2][y1:y2]
            
    
def train_nets():
    global  err, test_err, chromosomes, fitness
      
    for epoch in xrange(iter_no):       
                          
        for idx in xrange(chroms):             
            fitness[idx] = train_eval(chromosomes[idx])
        
        #print fitness        
        best = np.argmax(fitness)        
        err[epoch] = np.max(fitness)  
        test_err[epoch] = test_eval(chromosomes[best])
        
        pool()
        mutate()   
        crossover()                     
        
        if epoch%10 == 0:
            print "Iteration no: ", epoch, "    error: ", err[epoch], " test error: ", test_err[epoch]

    
def execute_net(inputs, synapses):
    #print inputs, synapses
    global receptors

    #activate the nodes based on sum of incoming synapses    
    receptors[0] = activate(synapses[0].dot(inputs)) #activate first time based on inputs
    for index in xrange(1,depth):        
        receptors[index] = activate(synapses[index].dot(receptors[index-1]))
        
    return receptors[depth-1]
     


def main():
    generate_chromosomes()
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
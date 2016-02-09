# -*- coding: utf-8 -*-
"""
Created on Tue Sep 08 06:49:35 2015

My First Neural Network :)

@author: yash
"""
#training_values = [[.01,0],[.04,0],[.05,0],[.09,0],[-.04,1], [-.03,1], [-.06,1]] #give some x,y values
training_values = [[.1,.1,0.75],[.4,.1,0.75],[.1,.4,0.75],[.1,.7,0],[.1,.9,0], [.3,.8,0], [.7,.7,0.25],[.6,.99,0.25],[.9,.7,0.25],[.8,.1,1],[.7,.3,1],[.99,2,1],[.5,.5,0.5],[.8,.5,0.5], [.4,.7,0.5], [.6,.8,0.5]]
learning_rate = .001
iterations = 100000 #training iteration count
#w = [0.3,0.1,0.05,0.09,0.4,0.008]#{'w1':0,'w2':0,'w3':0,'w4':0,'w5':0,'w6':0} #weights, try with random weights also
w = [0,0,0,0,0,0,0,0,0,0,0,0,0]
dw =  [0,0,0,0,0,0,0,0,0,0,0,0,0]#{'w1':0,'w2':0,'w3':0,'w4':0,'w5':0,'w6':0} #del_weights
e = 2.718281828459045
a,b,c,z,error = 0,0,0,0,0

for i in range(iterations):
    if i%10000 == 0:
        print '\n\n------------Iteration', i, '-----------'
    for item in training_values:
        #print '\n\nw: ', w
        #print 'dw: ', dw
        x,y, goal = item
        #print '\n'

        # Outputs of first hidden layer
        a = 1.0/(1+e**(-1*(x*w[0] + y*w[6] + w[9])))
        b = 1.0/(1+e**(-1*(x*w[1] + y*w[7] + w[10])))
        c = 1.0/(1+e**(-1*(x*w[2] + y*w[8] + w[11])))
        
        #print a,b,c
        #final output
        z_sum = a*w[3]+ b*w[4]+ c*w[5] + w[12]
        z = 1.0/(1+e**(-z_sum))
        
        #compute error_diff
        error = goal - z
        #print 'input: ', x, ' ', y , ' Expected: ', goal, '  output: ', z, '  error: ',error
        
        #backpropagate and update weights
        
        #weights connecting from hidden to output
        dw[5] = error*(z*(1-z))*a
        w[5] = w[5] + learning_rate*dw[5]

        dw[4] = error*(z*(1-z))*b
        w[4] = w[4] + learning_rate*dw[4]

        dw[3] = error*(z*(1-z))*c
        w[3] = w[3] + learning_rate*dw[3]
        

        #bias weights
        dw[12] = error*(z*(1-z))*1
        w[12] = w[12] + learning_rate*dw[12]

        dw[11] = error*(z*(1-z))*w[5]*(c*(1-c))*1
        w[11] = w[11] + learning_rate*dw[11]

        dw[10] = error*(z*(1-z))*w[4]*(b*(1-b))*1
        w[10] = w[10] + learning_rate*dw[10]        

        dw[9] = error*(z*(1-z))*w[3]*(a*(1-a))*1
        w[9] = w[9] + learning_rate*dw[9]

        #weights connected from x to hidden
        dw[2] = error*(z*(1-z))*w[5]*(c*(1-c))*x
        w[2] = w[2] + learning_rate*dw[2]

        dw[1] = error*(z*(1-z))*w[4]*(b*(1-b))*x
        w[1] = w[1] + learning_rate*dw[1]        

        dw[0] = error*(z*(1-z))*w[3]*(a*(1-a))*x
        w[0] = w[0] + learning_rate*dw[0]
        
        
        #weights connected from y to hidden
        dw[8] = error*(z*(1-z))*w[5]*(c*(1-c))*y
        w[8] = w[8] + learning_rate*dw[8]

        dw[7] = error*(z*(1-z))*w[4]*(b*(1-b))*y
        w[7] = w[7] + learning_rate*dw[7]        

        dw[6] = error*(z*(1-z))*w[3]*(a*(1-a))*y
        w[6] = w[6] + learning_rate*dw[6]
        
        #print 'w: ', w
        #print 'dw: ', dw
        
while(1):
    print 'classify your number: '
    x = input()
    if x == -999:
        break
    y = input()
    

    
    a = 1.0/(1+e**(-1*(x*w[0] + y*w[6])))
    b = 1.0/(1+e**(-1*(x*w[1] + y*w[7])))
    c = 1.0/(1+e**(-1*(x*w[2] + y*w[8])))
    
    #print a,b,c
    #final output
    z_sum = a*w[3]+ b*w[4]+ c*w[5]
    z = 1.0/(1+e**(-z_sum))
    
    print z
    
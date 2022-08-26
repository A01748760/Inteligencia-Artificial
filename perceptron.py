# Author: David Rodriguez Fragoso
# Date: 25/08/2022
# Program that creates a perceptron using the sign activation funcion and the gradient descent rule 

import numpy as np


def obtain_o(x,w):
    for i in range(len(w)):
       result = np.sign(np.dot(x[i],w[i]))  
    
    return result

def obtain_w(w,alpha,t,o,x):
    pass
    

def main():
    data = []
    weights = []
    with open("inputs.txt",'r') as file:
        for line in file:
            vector = line.strip().split(',')
            data.append(vector)
    
    #cast strings into floats
    for i in data:
        for j in range(0, len(i)):
            i[j] = float(i[j])
            if i[j] == 0.0:
                i[j] = -1

    #define learning rate, t, and vector x
    alpha = data[len(data)-1]
    t = data[len(data)-2]
    x = data[:-2]

    # set the initial weights to a value near to 0
    weights2 = [0.1 for i in range(0, len(x[0]))]
    
    for i in range (0, len(data)-2):
        weights.append(weights2)
    
    weights = np.array(weights).transpose()
    x = np.array(x).transpose()
    
    print(weights)
    print(x)

    obtain_o(x,weights)


main()
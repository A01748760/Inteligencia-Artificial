# Author: David Rodriguez Fragoso
# Date: 25/08/2022
# Program that creates a perceptron using the sign activation funcion and the gradient descent rule 

from turtle import pen
import numpy as np


def obtain_o(x,w):
    result = []
    for i in range(len(w)):
        result.append(np.sign(np.dot(x[i],w[i])))
    return result

def obtain_w(w,alpha,t,o,x):
    weights = []
    for i in range(len(w)):
        result = w[i]+alpha*(t[i]-o[i])*x[i]
        weights.append(result)
    return weights

    

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
    alpha = np.array(alpha)
    t = data[len(data)-2]
    x = data[:-2]

    # set the initial weights to a value near to 0
    weights2 = [0.1 for i in range(0, len(x[0]))]
    
    for i in range (0, len(data)-2):
        weights.append(weights2)
    
    weights = np.array(weights).transpose()
    #print(f'{"PESOS":=^60}\n {weights}')
    x = np.array(x).transpose()
    t = np.array(t).transpose()

    o = obtain_o(x,weights)
    o = np.array(o).transpose()


    w2 = obtain_w(weights,alpha,t,o,x)

    o = o.tolist()
    t = t.tolist()

    while o != t:
        weights = np.array(w2)
        t = np.array(t)
        o = obtain_o(x,weights)
        
        t = t.tolist()
    print(f'{"ENTRADAS":=^60}\n {x}')
    print(f'{"VALORES ESPERADOS":=^60}\n {t}')
    print(f'{"VALORES CALCULADOS":=^60}\n {o}')
    

main()
# Author: David Rodriguez Fragoso
# Date: 25/08/2022
# Program that creates a perceptron using the sign activation funcion and the gradient descent rule 

import random
from wave import Wave_write
import numpy as np


def obtain_o(x,w):
    result = []
    #print(f'{"x":=^60}')
    #print(x)
    for i in range(len(x)):
    #for i in range(x.shape[1]):
        result.append(np.sign(np.dot(x[i],w)))
    return result


def obtain_w(w,alpha,t,o,x):
    
    #print(f'w{w}')
    #print(f'o{o}')
    for i in range(len(x)):
        w = w+alpha*(t[i]-o[i])*x[i]


    return w

def getError(o, test):
    o = np.array(o)
    test = np.array(test)
    accuracy = (o == test).sum()/len(test)
    return accuracy*100

def main():
    #nTraining = int(input('Training data percentage: ')) 
    data = []
    weights = []
    with open("test4.txt",'r') as file:
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
    data.pop()
    alpha = np.array(alpha)
    
    #shuffle inputs
    #print(f'{"x":=^60}')
    data = np.array(data).transpose()
    np.random.shuffle(data)
    data = data.transpose()

    t = data[len(data)-1]
    x = data[:-1]
    

    # set the initial weights to a value near to 0
    weights = [0.1 for i in range(0, len(x))] 
    
    '''for i in range (0, len(data)-2):
        weights.append(weights2)'''
    weights = np.array(weights).transpose()
    #print(f'{"PESOS":=^60}\n {weights}')
    x = np.array(x)
    t = np.array(t)
    x = x.transpose()
    '''print(f't{t.shape}')
    print(f't matriz{t}')
    print(f'x{x.shape}')
    print(f'weights{weights.shape}')
    '''
    o = obtain_o(x,weights)
    o = np.array(o)
    
    #w2 = obtain_w(weights,alpha,t,o,x)
    
    
    o = o.tolist()
    t = t.tolist()

    dictW = dict()

    if str(weights) not in dictW:
        dictW[str(weights)]=1
    
    while o != t and dictW[str(weights)]<21:
        weights = np.array(obtain_w(weights,alpha,t,o,x))
        
        t = np.array(t)
        o = obtain_o(x,weights)
        
        
        t = t.tolist()
        o=np.array(o)
        o = o.tolist()
        if str(weights) not in dictW:
            dictW[str(weights)]=1
        else:
            dictW[str(weights)]+=1
    
        #print(weights)
    precision = getError(o,t)
    print(f'{"ENTRADAS":=^60}\n {x}')
    print(f'{"COEFICIENTE DE APRENDIZAJE":=^60}\n {alpha}')
    print(f'{"VALORES ESPERADOS":=^60}\n {t}')
    print(f'{"VALORES CALCULADOS":=^60}\n {o}')
    print(f'{"PESOS":=^60}\n {weights}')
    print(f'{"DICCIONARIO DE PESOS":=^60}\n{dictW}')
    print(f'{"PRECISIÓN":=^60}\n{precision}%')
    

main()
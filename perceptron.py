# Author: David Rodriguez Fragoso
# Date: 25/08/2022
# Program that creates a perceptron using the sign activation funcion and the gradient descent rule 

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def obtain_o(x,w):
    print(f"X:   {x}")
    print(f"W:   {w}")
    result = []
    #print(f'{"x":=^60}')
    #print(x)
    for i in range(x.shape[0]):
    #for i in range(x.shape[1]):
        result.append(np.sign(np.dot(x[i],w)))
        print(result)
    return result


def obtain_w(w,alpha,t,o,x):
    
    #print(f'w{w}')
    #print(f'o{o}')
    for i in range(len(x)):
        w = w+alpha*(t[i]-o[i])*x[i]


    return w

def getError(finalO, testT):
    return mean_squared_error(testT,finalO)

def graph(predict,real):
    plt.plot(real, label='Real')
    plt.plot(predict, label='Predction')
    plt.legend()
    plt.show()

def main():
    pTraining = int(input('Training data percentage: ')) 
    alpha = float(input("Alpha: "))
    data = []
    weights = []
    data = pd.read_csv('Test5.csv')
    data.replace(0,-1)
    
    pTraining = pTraining/100
    nTraining = int(np.floor(data.shape[0]*pTraining) )

    #shuffle data
    data = data.sample(frac=1).reset_index(drop=True)
    t = data.iloc[:,-1:]
    data = data.drop(t,axis=1)

    #split train and test data
    xTrain = data.iloc[:nTraining]
    xTest = data.iloc[nTraining:]

    xTrain = np.array(xTrain)
    xTest = np.array(xTest)
   
    # set the initial weights to a value near to 0
    weights = pd.Series([0.1] * (xTrain.shape[1]))
    weights = np.array(weights)
    t = np.array(t)
    
    o = obtain_o(xTrain,weights)
    o = np.array(o)
    print(o)
    
    o = o.tolist()
    t = t.tolist()

    dictW = dict()
   
    if str(weights) not in dictW:
        dictW[str(weights)]=1
    
    while o != t and dictW[str(weights)]<21:
        weights = np.array(obtain_w(weights,alpha,t,o,xTrain))
        
        t = np.array(t)
        o = obtain_o(xTrain,weights)
        
        
        t = t.tolist()
        o=np.array(o)
        o = o.tolist()
        if str(weights) not in dictW:
            dictW[str(weights)]=1
        else:
            dictW[str(weights)]+=1
    
    #print(testT)
    #print(list(dictW.keys())[-1])
    #oFinal = obtain_o(testT,list(dictW.keys())[-1])
    #print(oFinal)
    #precision = getError(oFinal,len(xTest)-1)
    print(f'{"ENTRADAS":=^60}\n {xTrain}')
    print(f'{"COEFICIENTE DE APRENDIZAJE":=^60}\n {alpha}')
    print(f'{"VALORES ESPERADOS":=^60}\n {t}')
    print(f'{"VALORES CALCULADOS":=^60}\n {o}')
    print(f'{"PESOS":=^60}\n {weights}')
    print(f'{"DICCIONARIO DE PESOS":=^60}\n{dictW}')
    #print(f'{"PRECISIÓN":=^60}\n{precision}%')
    graph(o,t)
    

main()
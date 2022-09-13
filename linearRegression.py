import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


data = pd.read_csv('winequality-red.csv')

dfX = data.iloc[:,:-1].values
dfY = data.iloc[:,-1:].values.ravel()

xTrain, xTest, yTrain, yTest = train_test_split(dfX,dfY, test_size=0.2, random_state=4)
perceptron = Perceptron()
perceptron.fit(xTrain, yTrain)
prediction = perceptron.predict(xTest)
print(prediction)

plt.plot(yTest, label='Real')
plt.plot(prediction, label='Predction')
plt.legend()
plt.show()

print(r2_score(yTest,prediction))
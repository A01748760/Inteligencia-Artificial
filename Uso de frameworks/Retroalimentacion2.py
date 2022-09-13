import statsmodels.api as sm
import pandas as pd
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv('wine.csv')

dfX = df[['flavanoids', 'malic acid', 'ash','magnesium','total phenols','alcalinity of ash']]
dfY = df['alcohol']

#Dividimos los datos en una proporcion 80/20
xTrain, xTest, yTrain, yTest = train_test_split(dfX,dfY,test_size=0.25, random_state=1)

#Agregamos la columna constante a ambos datasets
xTrain = sm.add_constant(xTrain)
xTest = sm.add_constant(xTest)

model = sm.OLS(yTrain, xTrain).fit()
# Ejecutamos el modelo entranado usando los datos de prueba
prediction = model.predict(xTest)
print(model.summary())

#Calculamos el MSE
print('MSE: ', mean_squared_error(yTest, prediction))

#Graficamos los resultados
plt.figure(figsize=(15, 15))
plt.plot(yTest.reset_index(drop=True), label='Valores reales')
plt.plot(prediction.reset_index(drop=True), label='Valores predichos')
plt.legend()
plt.xlabel('Valor real')
plt.ylabel('Prediccion')
plt.show()
from re import X
import pandas as pd
import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingClassifier
from mlxtend.evaluate import bias_variance_decomp
from sklearn.model_selection import train_test_split

df = pd.read_csv('wine.csv')

dfX = df[['flavanoids', 'malic acid', 'ash','magnesium','total phenols','alcalinity of ash','OD280/OD315 of diluted wines','proline','hue','color intensity']]
dfY = df['alcohol']

#Dividimos los datos en una proporcion 75/25
xTrain, xTest, yTrain, yTest = train_test_split(dfX,dfY,test_size=0.3, random_state=1)

model = LinearRegression()

# Ejecutamos el modelo entranado usando los datos de prueba
# Calculamos porcentajes de bias y varianza
mse, bias, var = bias_variance_decomp(model, xTrain.values, yTrain.values, xTest.values, yTest.values, loss='mse', num_rounds=200, random_seed=1)
prediction = model.predict(xTest.values)
# summarize results
print('MSE: %.3f' % mse)
print('Bias: %.3f' % bias)
print('Variance: %.3f' % var)

#Graficamos los resultados
plt.figure(figsize=(15, 15))
plt.plot(yTest.values, label='Valores reales')
plt.plot(prediction, label='Valores predichos')
plt.legend()
plt.xlabel('Valor real')
plt.ylabel('Prediccion')
plt.show()
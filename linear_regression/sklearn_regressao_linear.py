# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 14:41:28 2018

@author: USER_
"""

# ----------------------------------------------------
#           SKLEARN - REGRESSAO LINEAR MULTIPLA
# ----------------------------------------------------

#importar dataset
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('dataset/housing.data.txt', header = None, sep = '\s+')
dataset.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
                   'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

x = dataset.iloc[:, :-1].values
y = dataset['MEDV'].values              

#preprocessamento: standarizacao
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x = sc_x.fit_transform(x)
sc_y = StandardScaler() 
y = sc_y.fit_transform(y)          
           
#dividir conjunto de treinamento - conjunto de teste
from sklearn.model_selection import train_test_split           
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                            test_size = 0.3, random_state = 0)
           
#fase de treinamento
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#predicao de valores 
y_pred_train = regressor.predict(x_train)
y_pred_test = regressor.predict(x_test)

#gráfica de residuos
plt.scatter(y_pred_train, y_pred_train - y_train, c = 'blue', 
            marker = 'o', label = 'training data')
plt.scatter(y_pred_test, y_pred_test - y_test, c = 'lightgreen', 
            marker = 's', label = 'test data')
plt.xlabel('valores preditos')
plt.ylabel('residuos')
plt.legend(loc = 'upper left')
plt.hlines(y = 0, xmin = -10, xmax = 30, color = 'red', linestyles = 'solid')
plt.xlim(-10, 30)
plt.show()

#calculo do erro: MSE [mean-squared-error]
from sklearn.metrics import mean_squared_error
print('MSE - train: ', round(mean_squared_error(y_train, y_pred_train), 3))
print('MSE - test: ', round(mean_squared_error(y_test, y_pred_test), 3))

#calculo da determinacao: R^2 [R-squared]
from sklearn.metrics import r2_score
print('R^2 - train: ', round(r2_score(y_train, y_pred_train), 3))
print('R^2 - test: ', round(r2_score(y_test, y_pred_test), 3))

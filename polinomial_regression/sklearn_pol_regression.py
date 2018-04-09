# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 17:25:23 2018

@author: USER_
"""

# ------------------------------------------------
#              REGRESSAO POLINOMIAL
# ------------------------------------------------

#importar dataset
import pandas as pd
import numpy as np
dataset = pd.read_csv('dataset/housing.data.txt', header = None, sep = '\s+')
dataset.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
                   'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

#definicao de variaveis
x = dataset[['LSTAT']].values
y = dataset['MEDV'].values

#definicao do regressor 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#preprocessamento
from sklearn.preprocessing import PolynomialFeatures

#termo quadratico
quadratico = PolynomialFeatures(degree = 2)
x_2 = quadratico.fit_transform(x)

#termo cubico
cubico = PolynomialFeatures(degree = 3)
x_3 = cubico.fit_transform(x)

#dominio de X
x_fit = np.arange(x.min(), x.max(), 1)[:, np.newaxis]

from sklearn.metrics import r2_score

#regressao linear
regressor = regressor.fit(x, y)
y_lin_fit = regressor.predict(x_fit)
y_pred_lin = regressor.predict(x)
r2_lin = round(r2_score(y, y_pred_lin), 3)

#regressao quadratica
regressor = regressor.fit(x_2, y)
y_quad_fit = regressor.predict(quadratico.fit_transform(x_fit))
y_pred_quad =regressor.predict(x_2)
r2_quad = round(r2_score(y, y_pred_quad), 3)

#regressao cubica
regressor = regressor.fit(x_3, y)
y_cub_fit = regressor.predict(cubico.fit_transform(x_fit))
y_pred_cub = regressor.predict(x_3)
r2_cub = round(r2_score(y, y_pred_cub), 3)

#gráfica de resultados
import matplotlib.pyplot as plt
plt.scatter(x, y, label = 'train', color = 'lightgray')
#modelo linear
plt.plot(x_fit, y_lin_fit, label = 'linear: $R^2 = %.2f$'%r2_lin, color = 'blue')
plt.plot(x_fit, y_quad_fit, label = 'quad: $R^2 = %.2f$'%r2_quad, color = 'red')
plt.plot(x_fit, y_cub_fit, label = 'cub: $R^2 = %.2f$'%r2_cub, color = 'green')
plt.title('Regressao polinomial')
plt.xlabel('% populacao baixos recursos [LSTAT]')
plt.ylabel('Preco em $1000 [MEDV]')
plt.legend(loc = 'upper left')
plt.show()


           
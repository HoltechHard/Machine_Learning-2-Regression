# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 13:22:40 2018

@author: USER_
"""

# --------------------------------------------------------
#               REGRESSAO LINEAR MULTIPLA                
# --------------------------------------------------------

#classe regressao linear

import numpy as np

class LinearRegression(object):
    
    def __init__(self, eta = 0.001, n_iter = 25):
        self.eta = eta
        self.n_iter = n_iter
    
    def inicialize(self, x):
        self.w = np.zeros(1 + x.shape[1])
        self.cost = []
    
    def fit(self, x, y):
        
        self.inicialize(x)
        
        for i in range(self.n_iter):
            y_pred = self.input_net(x)
            errors = y - y_pred
            
            self.w[1:] += self.eta * np.dot(x.T, errors)
            self.w[0]  += self.eta * errors.sum()
            
            mse = sum(errors **2)/2.0
            
            self.cost.append(mse)
        
        return self
    
    def input_net(self, x):
        return np.dot(x, self.w[1:]) + self.w[0]
    
    def predict(self, x):
        return self.input_net(x)
    
    
#importar dataset
import pandas as pd
dataset = pd.read_csv('dataset/housing.data.txt', header = None, sep = '\s+')
dataset.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
                   'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

#analise exploratorio das variaveis
import matplotlib.pyplot as plt
import seaborn as sbn
sbn.set(context = 'notebook', style = 'darkgrid')
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sbn.pairplot(dataset[cols], size = 2.0)
plt.show()

#analise de correlacao
tbl_corr = np.corrcoef(dataset[cols].values.T)
sbn.set(font_scale = 1.0)
h_map = sbn.heatmap(tbl_corr, cbar = True, annot = True, square = True, 
                    fmt = '.2f', annot_kws = {'size' : 15}, 
                    yticklabels = cols, xticklabels = cols)
plt.title('Analise correlacao')
plt.show()


#preprocessamento
x = dataset.iloc[:, :-1].values
y = dataset['MEDV'].values                

#standarizacao dos dados
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x = sc_x.fit_transform(x)

sc_y = StandardScaler()
y = sc_y.fit_transform(y)           
           
from sklearn.model_selection import train_test_split           
x_train, x_test, y_train, y_test = train_test_split(x, y,
                        test_size = 0.3, random_state = 0)


#treinamento - regressao linear multipla
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#predicao de valores
y_pred_train = regressor.predict(x_train)
y_pred_test  = regressor.predict(x_test) 

#gráfica de dispersao
plt.scatter(y_pred_train, y_pred_train - y_train, c = 'blue', 
            marker = 's', label = 'training data')
plt.scatter(y_pred_test, y_pred_test - y_test, c = 'lightgreen', 
            marker = 's', label = 'test data')

plt.xlabel('Valores preditos')
plt.ylabel('Residuos')
plt.legend(loc = 'upper left')
plt.hlines(y = 0, xmin = -10, xmax = 30, lw = 2, color = 'red')
plt.xlim([-10, 30])
plt.show()

from sklearn.metrics import mean_squared_error
print('MSE train: ', round(mean_squared_error(y_train, y_pred_train), 3))
print('MSE test: ', round(mean_squared_error(y_test, y_pred_test), 3))

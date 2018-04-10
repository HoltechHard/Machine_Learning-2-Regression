# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 22:53:02 2018

@author: USER_
"""

# ------------------------------------------------------
#              ÁRVORES DE DECISAO DE REGRESSAO
# ------------------------------------------------------

import pandas as pd

#importar dataset
dataset = pd.read_csv('dataset/housing.data.txt', header = None, sep = '\s+')
dataset.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
                   'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
x = dataset[['LSTAT']].values
y = dataset['MEDV'].values           

#dividir conjunto de treinamento - conjunto de teste
from sklearn.model_selection import train_test_split           
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                            test_size = 0.25, random_state = 0)           

#fase de treinamento
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(criterion = 'mse', max_depth = 3)
tree.fit(x_train, y_train)

#predicao de resultados
y_pred_train = tree.predict(x_train)
y_pred_test = tree.predict(x_test)

#validacao do modelo
from sklearn.metrics import r2_score
r2_train = round(r2_score(y_train, y_pred_train), 3)
r2_test = round(r2_score(y_test, y_pred_test), 3)

#grafica do modelo de regressao
import matplotlib.pyplot as plt

#argsort(): funcao de numpy que devolve os indices de um vetor ordenado
#flatten(): funcao de numpy que unifica todos os dados em 1 dimensao

#funcao para graficar
def graph_regressor(x, y, model, lbl, r2):
    plt.scatter(x, y, color = 'blue', label = lbl)
    plt.title('Arvore de decisao de regressao')
    plt.xlabel('% populacao baixos recursos [LSTAT]')
    plt.ylabel('Preco em $1000 [MEDV]')
    plt.plot(x, model.predict(x), color = 'red', label = 'tree: r^2 = ' + str(r2))
    plt.legend(loc = 'upper left')
    plt.show()

#grafica para dados de treinamento
idx_train = x_train.flatten().argsort()        
graph_regressor(x_train[idx_train], y_train[idx_train], tree, 'train', r2_train)

#grafica para dados de teste
idx_test = x_test.flatten().argsort()
graph_regressor(x_test[idx_test], y_test[idx_test], tree, 'test', r2_test)

           

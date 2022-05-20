# -*- coding: utf-8 -*-
"""
Created on Tue May 17 17:49:40 2022

@author: Ja
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import pickle
import requests


iris=load_iris()
iris['data']

df = pd.DataFrame(data = np.c_[iris['data'], iris['target']], 
                  columns=iris['feature_names']+['target'])

class Perceptron:
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        #print(self.w_)
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                #print(y)
                update=self.eta*(target-self.predict(xi))
                update_xi = update[0,0]*xi
                update_xi = np.array(update_xi).flatten()
                #print(update_xi)
                #print(self.w_[1:])
                self.w_[1:] += update_xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    
    def net_input(self, X):
        #print(np.dot(X, self.w_[1:]) + self.w_[0])
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        #print(np.where(self.net_input(X) >= 0, 1, -1))
        return np.where(self.net_input(X) >= 0, 1, -1)

    
    
mod = Perceptron()
X_iris = np.matrix(df.iloc[:,:4])
y_iris = np.matrix(df.iloc[:,4]).reshape(150,1)
#print(y_iris)
mod.fit(X_iris, y_iris)

pickle.dump(mod, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([1.,1.,2.,1.]))

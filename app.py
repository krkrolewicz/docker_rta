# -*- coding: utf-8 -*-
"""
Created on Tue May 17 17:58:30 2022

@author: Ja
"""

import numpy as np
#from joblib import load
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import pickle
from model import Perceptron


app = Flask(__name__)
api = Api(app)

model_s = pickle.load(open("model.pkl",'rb'))


@app.route('/', methods = ['GET'])
def h():
    return("Witamy! Rozszerzenie linku do predykcji: /api/predict")

@app.route('/api/predict', methods=['GET'])
def get_prediction():
    sepal_length = float(request.args.get('sl', "1"))
    print(sepal_length)
    sepal_width = float(request.args.get('sw', "1"))
    petal_length = float(request.args.get('pl', "1"))
    petal_width = float(request.args.get('pw', "1"))
    
    features = [sepal_length, sepal_width, petal_length, petal_width]
    #model = mod#load('model.joblib')
    predicted_class = int(model_s.predict(features)) #int(model.predict(features))

    return jsonify(features=features, predicted_class=predicted_class)


if __name__ == '__main__':
    #app.debug = True
    app.run(host='0.0.0.0')
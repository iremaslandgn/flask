# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 22:01:12 2019

@author: irem
"""
from flask import Flask,request, jsonify, render_template
import pickle


app = Flask(__name__)

a = pickle.load(open("model.pkl","rb"))

@app.route('/predict', methods=["POST"])
def predict():
    
    global a
    """data = request.form"""
    req_data = request.get_json()
    data = req_data
    data1 = req_data["data_1"]
    data2 = req_data["data_2"]
    data3 = req_data["data_3"]
    data4 = req_data["data_4"]
    data5 = req_data["data_5"]

    datakredi=float(data1)
    datakredi=(datakredi-250)/18174
    
    datayas=float(data2)
    datayas=(datayas-19)/75
    
    datakredisayisi=float(data4)
    datakredisayisi=(datakredisayisi-1)/4
    
      
    data = [datakredi, datayas, data3, datakredisayisi, data5 ]
    print(data)
    res = list(a.predict([data]))
    return jsonify([res])

@app.route('/', methods=["GET"])
def home():
    # arayuzu koddan ayiralim
    return render_template("home_template.html")

app.run()
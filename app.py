from flask import Flask,request, url_for, redirect, render_template, jsonify
import xgboost
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('XGB_Alquiler.sav', 'rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    final_features = pd.DataFrame(final_features)
    final_features.columns = ['barrio_cat', 'ambientes', 'm2']
    prediction = model.predict(final_features)

    output = prediction
    return render_template('index.html',prediction_text='Alquiler estimado {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data = pd.DataFrame.from_dict(data)

    prediction = model.predict(data)

    output = prediction
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)

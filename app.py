import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
import joblib
import requests

from utils import Logger

app=Flask(__name__)

from sklearn.base import BaseEstimator, TransformerMixin


# class Logger(BaseEstimator, TransformerMixin):
#     def __init__(self, apply_log = True):
#         self.apply_log = apply_log
        
#     def fit(self, X, y=None):
#         return self
    
#     def transform(self, X, y=None):
#         logX = X.copy()
        
#         if self.apply_log:
#             logX = np.log1p(X)
#             return logX
    
#         else: return X

## Load the model
cancerDetection = joblib.load('cancerDetectionStakingModel_joblib')

scalar= joblib.load('std_sca_joblib')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/benigno')
def benigno():
    return render_template('benigno.html')

@app.route('/maligno')
def maligno():
    return render_template('maligno.html')


@app.route('/predict',methods=['POST'])
def predict():
    print(request.form.values())
    data=[float(x) for x in request.form.values()]


    columns = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
       'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'radius_se', 'texture_se', 'perimeter_se',
       'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
       'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst', 'concavity_worst',
       'concave points_worst', 'symmetry_worst']

    tuned_names = ['Logistic Regression', 'SVC', 'GradientBoosting', 'RandomForest', 'KNNeighbors']

    logger = Logger()
    
    logreg_tuned = joblib.load('logreg_tuned_joblib')
    std_sca = joblib.load('std_sca_joblib')
    svc_tuned = joblib.load('svc_tuned_joblib')
    gdb_tuned = joblib.load('gdb_tuned_joblib')
    rf_tuned = joblib.load('rf_tuned_joblib')
    knn_tuned = joblib.load('knn_tuned_joblib')

    tuned_test_pred = np.zeros(shape=(1, 5)) # 5 models

    test = pd.DataFrame(data=[data], columns=columns)

    tuned_test_pred[:,0] = logreg_tuned.predict_proba(test)[:,1]
    tuned_test_pred[:,1] = svc_tuned.predict_proba(test)[:,1]
    tuned_test_pred[:,2] = gdb_tuned.predict_proba(test)[:,1]
    tuned_test_pred[:,3] = rf_tuned.predict_proba(test)[:,1]
    tuned_test_pred[:,4] = knn_tuned.predict_proba(test)[:,1]


    tuned_test_pred = pd.DataFrame(data=tuned_test_pred, 
                           columns=tuned_names)  

    X_test_scaled = std_sca.transform(test)

    X_test_final = np.concatenate([X_test_scaled, tuned_test_pred], axis=1)
    
    request_body = {
    "Input": X_test_final.tolist()
    }

    print(request_body)
    data = json.loads(json.dumps(request_body))
    payload = json.dumps(data)

    res = requests.post('https://6ua8sc2zr3.execute-api.us-east-1.amazonaws.com/PROD/api-ml-model', data = payload)

    if(res.content == b'"B"'):
        finalLabel = "TUMOR BENIGNO"
    else:
        finalLabel = "TUMOR MALIGNO"

    return render_template("home.html",prediction_text="La prediccion final es: {}".format(finalLabel))



if __name__=="__main__":
    app.run(debug=False)
   
     

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import joblib


class Logger(BaseEstimator, TransformerMixin):
    def __init__(self, apply_log = True):
        self.apply_log = apply_log
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        logX = X.copy()
        
        if self.apply_log:
            logX = np.log1p(X)
            return logX
    
        else: return X
        
# @app.before_first_request
# def load_models():
#     global logreg_tuned
#     global std_sca
#     global svc_tuned
#     global gdb_tuned
#     global rf_tuned
#     global knn_tuned


#     logreg_tuned = joblib.load('logreg_tuned_joblib')
#     std_sca = joblib.load('std_sca_joblib')
#     svc_tuned = joblib.load('svc_tuned_joblib')
#     gdb_tuned = joblib.load('gdb_tuned_joblib')
#     rf_tuned = joblib.load('rf_tuned_joblib')
#     knn_tuned = joblib.load('knn_tuned_joblib')
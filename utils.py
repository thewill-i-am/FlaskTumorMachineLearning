from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import joblib

global Logger 

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
        
Logger = Logger
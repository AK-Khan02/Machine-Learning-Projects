import numpy as np
import pandas as pd

class SimpleImputer:
    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.statistics_ = None

    def fit(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        if self.strategy == 'mean':
            self.statistics_ = X.mean()
        elif self.strategy == 'median':
            self.statistics_ = X.median()
        else:
            raise ValueError("Strategy not recognized: should be 'mean' or 'median'")

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X.fillna(self.statistics_)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

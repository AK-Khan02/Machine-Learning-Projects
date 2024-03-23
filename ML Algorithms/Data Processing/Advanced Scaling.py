import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy.stats import skew

class AdvancedScaler:
    def __init__(self):
        self.scalers = {}
        self.log_transform_features = []

    def fit(self, X):
        """
        Determine which scaler to apply to each feature based on the data distribution.
        """
        for column in X.columns:
            data = X[column]
            
            # Determine if log transformation is needed (high skewness)
            if skew(data) > 0.75:
                data = np.log1p(data)
                self.log_transform_features.append(column)
            
            # Decide on the scaling strategy
            if data.min() < 0 or skew(data) > 0.75:
                self.scalers[column] = StandardScaler()
            elif len(np.unique(data)) < 10:
                self.scalers[column] = MinMaxScaler(feature_range=(0, 1))
            else:
                self.scalers[column] = RobustScaler()
            
            # Fit the chosen scaler
            self.scalers[column].fit(data.values.reshape(-1, 1))

    def transform(self, X):
        """
        Apply the determined scaler to each feature.
        """
        X_scaled = pd.DataFrame()
        for column in X.columns:
            data = X[column]
            
            # Apply log transformation if previously determined
            if column in self.log_transform_features:
                data = np.log1p(data)
            
            # Apply the fitted scaler
            scaled_data = self.scalers[column].transform(data.values.reshape(-1, 1))
            X_scaled[column] = scaled_data.flatten()
        
        return X_scaled

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

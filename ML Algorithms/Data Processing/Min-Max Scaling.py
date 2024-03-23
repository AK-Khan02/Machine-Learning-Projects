import numpy as np

class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.scale_ = None

    def fit(self, X):
        self.min_ = np.min(X, axis=0)
        data_range = np.max(X, axis=0) - self.min_
        range_min, range_max = self.feature_range
        self.scale_ = (range_max - range_min) / data_range

    def transform(self, X):
        X_std = (X - self.min_) * self.scale_
        range_min, _ = self.feature_range
        return X_std + range_min

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# Sample data
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)

# Min-Max Scaling
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = min_max_scaler.fit_transform(data)
print("Data after Min-Max Scaling:\n", data_normalized)

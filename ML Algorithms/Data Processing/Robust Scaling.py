class RobustScaler:
    def __init__(self):
        self.center_ = None
        self.scale_ = None

    def fit(self, X):
        self.center_ = np.median(X, axis=0)
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        self.scale_ = Q3 - Q1

    def transform(self, X):
        return (X - self.center_) / self.scale_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# Sample data with outliers
data_with_outliers = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [100, 200, 300]], dtype=float)

# Robust Scaling
robust_scaler = RobustScaler()
data_robust_scaled = robust_scaler.fit_transform(data_with_outliers)
print("Data after Robust Scaling:\n", data_robust_scaled)

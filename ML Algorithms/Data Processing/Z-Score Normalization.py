class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0, ddof=0)

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# Z-Score Standardization
standard_scaler = StandardScaler()
data_standardized = standard_scaler.fit_transform(data)
print("\nData after Z-Score Standardization:\n", data_standardized)

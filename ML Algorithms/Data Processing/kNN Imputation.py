from sklearn.impute import KNNImputer

class KNNImputerWrapper:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.imputer = KNNImputer(n_neighbors=n_neighbors)

    def fit(self, X):
        self.imputer.fit(X)

    def transform(self, X):
        return self.imputer.transform(X)

    def fit_transform(self, X):
        return self.imputer.fit_transform(X)

# k-NN Imputation
knn_imputer = KNNImputerWrapper(n_neighbors=2)
data_knn_imputed = knn_imputer.fit_transform(data)
print("\nData after k-NN Imputation:\n", data_knn_imputed)

from sklearn.tree import DecisionTreeRegressor

class GradientBoosting(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.tree_weights = []

    def fit(self, X, y):
        # Initial prediction is the mean of the target values
        f0 = np.mean(y)
        self.initial_prediction = f0
        Fm = np.full(len(y), f0)
        
        for _ in range(self.n_estimators):
            # Compute pseudo-residuals
            residuals = y - Fm
            
            # Fit a base learner
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            
            # Compute the multiplier for the tree's predictions
            predictions = tree.predict(X)
            multiplier = np.sum(residuals * predictions) / np.sum(predictions ** 2)
            
            # Update the model
            Fm += self.learning_rate * multiplier * predictions
            self.trees.append(tree)
            self.tree_weights.append(self.learning_rate * multiplier)

    def predict(self, X):
        Fm = np.full(X.shape[0], self.initial_prediction)
        for tree, weight in zip(self.trees, self.tree_weights):
            Fm += weight * tree.predict(X)
        return Fm
